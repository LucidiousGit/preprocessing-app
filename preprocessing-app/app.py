# pip installs:
# pip install pandas numpy pyqt5 scikit-learn

# due to the complexity of the script, some parts of the code have been written with the help of AI generation


import sys
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from main_window import Ui_DataPrep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

class MyApp(QtWidgets.QMainWindow, Ui_DataPrep):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.browseFiles.clicked.connect(self.select_file)
        
        self.saveButton.clicked.connect(self.save_file)

        self.chooseSaveLocation.clicked.connect(self.select_save_location)

        self.handleMissingValuesColumnList.itemClicked.connect(self.column_clicked)
        self.lagFeaturesEngineerFeaturesColumnList.itemClicked.connect(self.lag_column_clicked)
        self.rollingStatisticsEngineerFeaturesColumnList.itemClicked.connect(self.rolling_column_clicked)
        self.cumulativeStatisticsEngineerFeaturesColumnList.itemClicked.connect(self.cumulative_column_clicked)
        self.settingsColumnList.itemClicked.connect(self.settings_column_clicked)
        self.selectFeaturesColumnList.itemClicked.connect(self.feature_selection_column_clicked)
        self.selectFeaturesOptionDropdown.currentIndexChanged.connect(self.feature_selection_method_changed)

        self.strategyDropdown.currentIndexChanged.connect(self.strategy_changed)
        
        self.handleMissingValuesApplyButton.clicked.connect(self.apply_strategy)
        
        self.engineerFeaturesDropdown.currentIndexChanged.connect(self.engineer_features_changed)
        
        self.engineerFeaturesApplyButton.clicked.connect(self.apply_feature_engineering)
        
        self.applyScalingButton.clicked.connect(self.apply_scaling)

        self.applyCategoricalEncodingCheckbox.stateChanged.connect(self.toggle_categorical_encoding)
        self.applyNumericalScalingCheckbox.stateChanged.connect(self.toggle_scaling_options)

        self.selectAsNumericalButton.clicked.connect(self.set_as_numerical)
        self.selectAsCategoricalButton.clicked.connect(self.set_as_categorical)
        self.selectAsDatetimeButton.clicked.connect(self.set_as_datetime)

        self.selectFeaturesOptionDropdown.currentIndexChanged.connect(self.feature_selection_method_changed)
        self.selectFeaturesDeleteColumnButton.clicked.connect(self.delete_selected_feature_column)
        self.applySelectFeaturesButton.clicked.connect(self.apply_feature_selection)
        self.selectTargetColumnButton.clicked.connect(self.select_target_column)
        
        self.df = pd.DataFrame()
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.test_size = 0.2
        self.random_state = 42
        self.selected_datetime_column = None
        self.target_column = None
        self.save_folder_path = None

        self.testSizeValueEdit.setValidator(QDoubleValidator(0.01, 0.99, 2))
        self.testSizeValueEdit.setText("0.2")
        self.randomStateValueEdit.setValidator(QIntValidator(0, 9999))
        self.randomStateValueEdit.setText("42")

        self.testSizeValueEdit.textChanged.connect(self.update_test_size)
        self.randomStateValueEdit.textChanged.connect(self.update_random_state)

        self.hide_all_widgets()
        self.csvFilePreview.hide()
        self.errorLabel.hide()

    def hide_all_widgets(self):
        self.hide_preprocessing_widgets()
        self.hide_settings_widgets()
        self.hide_feature_selection_widgets()

    def hide_preprocessing_widgets(self):
        self.handleMissingValuesColumnList.hide()
        self.missingCountLabel.hide()
        self.strategyDropdown.hide()
        self.specificValueEdit.hide()
        self.handleMissingValuesApplyButton.hide()
        self.engineerFeaturesDropdown.hide()
        self.optionsStackedWidget.hide()
        self.engineerFeaturesApplyButton.hide()
        self.numericalFeatureScalingOptionDropdown.hide()
        self.applyScalingButton.hide()
        self.chooseNCDColumnsLabel.hide()
        self.applyCategoricalEncodingCheckbox.hide()
        self.applyNumericalScalingCheckbox.hide()

    def hide_settings_widgets(self):
        self.settingsColumnList.hide()
        self.selectAsCategoricalButton.hide()
        self.selectAsNumericalButton.hide()
        self.selectAsDatetimeButton.hide()
        self.testSizeLabel.hide()
        self.testSizeValueEdit.hide()
        self.randomStateLabel.hide()
        self.randomStateValueEdit.hide()

    def show_settings_widgets(self):
        self.settingsColumnList.show()
        self.selectAsCategoricalButton.show()
        self.selectAsNumericalButton.show()
        self.selectAsDatetimeButton.show()
        self.testSizeValueEdit.show()
        self.randomStateValueEdit.show()
        self.testSizeLabel.show()
        self.randomStateLabel.show()
        self.chooseNCDColumnsLabel.show()
        self.applyCategoricalEncodingCheckbox.show()
        self.applyNumericalScalingCheckbox.show()

    def hide_feature_selection_widgets(self):
        self.selectTargetColumnButton.hide()
        self.selectFeaturesColumnList.hide()
        self.selectFeaturesOptionDropdown.hide()
        self.selectFeaturesDeleteColumnButton.hide()
        self.correlationCoefficientThresholdLabel.hide()
        self.correlationCoefficientThresholdValueEdit.hide()
        self.applySelectFeaturesButton.hide()

    def show_feature_selection_widgets(self):
        self.selectTargetColumnButton.show()
        self.selectFeaturesColumnList.show()
        self.selectFeaturesOptionDropdown.show()
        self.selectFeaturesDeleteColumnButton.show()
        self.correlationCoefficientThresholdLabel.show()
        self.correlationCoefficientThresholdValueEdit.show()
        self.applySelectFeaturesButton.show()

    def show_list_widget(self):
        self.handleMissingValuesColumnList.show()

    def show_column_widgets(self):
        self.missingCountLabel.show()
        self.strategyDropdown.show()
        self.handleMissingValuesApplyButton.show()

    def show_engineer_features_widgets(self):
        self.engineerFeaturesDropdown.show()
        self.optionsStackedWidget.show()
        self.engineerFeaturesApplyButton.show()

    def show_scaling_widgets(self):
        self.applyScalingButton.show()
        self.numericalFeatureScalingOptionDropdown.show()

    def strategy_changed(self):
        if self.strategyDropdown.currentText() == "Fill with specific value":
            self.set_input_validator(self.selected_column)
            self.specificValueEdit.show()
        else:
            self.specificValueEdit.hide()

    def toggle_categorical_encoding(self):
        pass

    def toggle_scaling_options(self):
        if self.applyNumericalScalingCheckbox.isChecked():
            self.numericalFeatureScalingOptionDropdown.show()
            self.applyScalingButton.show()
        else:
            self.numericalFeatureScalingOptionDropdown.hide()
            self.applyScalingButton.show()

    def select_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.rawFilePath.setText(file_path)
            self.load_csv(file_path)
    
    def load_csv(self, file_path):
        self.df = pd.read_csv(file_path, dtype=self.get_dtypes(file_path))
        self.update_table_view()
        self.populate_column_list()
        self.readMeLabel.hide()
        self.csvFilePreview.show()
        self.show_list_widget()
        self.show_engineer_features_widgets()
        self.show_scaling_widgets()
        self.show_settings_widgets()
        self.show_feature_selection_widgets()

    def get_dtypes(self, file_path):
        sample_df = pd.read_csv(file_path, nrows=100)
        dtypes = {}
        for column in sample_df.columns:
            if pd.api.types.is_numeric_dtype(sample_df[column]):
                if sample_df[column].dropna().apply(lambda x: isinstance(x, (int, float)) and (x == int(x))).all():
                    dtypes[column] = 'Int64'
                else:
                    dtypes[column] = 'float'
            else:
                dtypes[column] = 'object'
        return dtypes

    def update_table_view(self):
        model = self.create_table_model(self.df)
        self.csvFilePreview.setModel(model)
    
    def create_table_model(self, df):
        model = QtGui.QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels(df.columns)
        for row in df.itertuples():
            for column, value in enumerate(row[1:]):
                item = QtGui.QStandardItem(str(value))
                model.setItem(row.Index, column, item)
        return model
    
    def populate_column_list(self):
        self.handleMissingValuesColumnList.clear()
        self.lagFeaturesEngineerFeaturesColumnList.clear()
        self.rollingStatisticsEngineerFeaturesColumnList.clear()
        self.cumulativeStatisticsEngineerFeaturesColumnList.clear()
        self.settingsColumnList.clear()
        self.selectFeaturesColumnList.clear()

        columns = self.df.columns.tolist()
        if self.target_column in columns:
            columns.remove(self.target_column)

        missing_value_columns = [col for col in columns if self.df[col].isnull().any()]
        self.handleMissingValuesColumnList.addItems(missing_value_columns)
        self.lagFeaturesEngineerFeaturesColumnList.addItems(columns)
        self.rollingStatisticsEngineerFeaturesColumnList.addItems(columns)
        self.cumulativeStatisticsEngineerFeaturesColumnList.addItems(columns)
        self.settingsColumnList.addItems(columns + [self.target_column])
        self.selectFeaturesColumnList.addItems(columns)
        self.update_column_list()

    def update_column_list(self):
        def update_list_widget(widget, columns):
            widget.clear()
            for column in columns:
                if column in self.datetime_columns:
                    item_text = f"{column} - datetime"
                elif column in self.numeric_columns:
                    item_text = f"{column} - numerical"
                elif column in self.categorical_columns:
                    item_text = f"{column} - categorical"
                else:
                    item_text = column
                widget.addItem(item_text)
            if self.target_column:
                widget.addItem(f"{self.target_column} - target")
        
        update_list_widget(self.handleMissingValuesColumnList, self.df.columns)
        update_list_widget(self.lagFeaturesEngineerFeaturesColumnList, self.df.columns)
        update_list_widget(self.rollingStatisticsEngineerFeaturesColumnList, self.df.columns)
        update_list_widget(self.cumulativeStatisticsEngineerFeaturesColumnList, self.df.columns)
        update_list_widget(self.settingsColumnList, self.df.columns)
        update_list_widget(self.selectFeaturesColumnList, self.df.columns)

    def settings_are_valid(self):
        if not all([col in self.numeric_columns + self.categorical_columns + self.datetime_columns for col in self.df.columns]):
            return False

        try:
            test_size = float(self.testSizeValueEdit.text())
            if not (0.01 <= test_size <= 0.99):
                return False
        except ValueError:
            return False

        try:
            random_state = int(self.randomStateValueEdit.text())
            if not (0 <= random_state <= 9999):
                return False
        except ValueError:
            return False

        return True

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.navigationBar.currentChanged.connect(self.on_tab_change)

    def on_tab_change(self, index):
        if index != self.navigationBar.indexOf(self.settings):
            if not self.settings_are_valid():
                QtWidgets.QMessageBox.warning(self, "Invalid Settings", "Please ensure all columns have a type, and valid test size and random state values are set.")
                self.navigationBar.setCurrentIndex(self.navigationBar.indexOf(self.settings))

    def settings_column_clicked(self, item):
        self.selected_column = item.text().split(" - ")[0]

    def set_as_numerical(self):
        column_name = self.selected_column
        if column_name:
            if column_name not in self.numeric_columns:
                self.numeric_columns.append(column_name)
            if column_name in self.categorical_columns:
                self.categorical_columns.remove(column_name)
            if column_name in self.datetime_columns:
                self.datetime_columns.remove(column_name)
            self.update_column_list()

    def set_as_categorical(self):
        column_name = self.selected_column
        if column_name:
            if column_name not in self.categorical_columns:
                self.categorical_columns.append(column_name)
            if column_name in self.numeric_columns:
                self.numeric_columns.remove(column_name)
            if column_name in self.datetime_columns:
                self.datetime_columns.remove(column_name)
            self.update_column_list()

    def set_as_datetime(self):
        column_name = self.selected_column
        if column_name:
            if column_name not in self.datetime_columns:
                self.datetime_columns.append(column_name)
            if column_name in self.numeric_columns:
                self.numeric_columns.remove(column_name)
            if column_name in self.categorical_columns:
                self.categorical_columns.remove(column_name)
            self.update_column_list()
            self.selected_datetime_column = column_name

    def column_clicked(self, item): 
        self.selected_column = item.text().split(" - ")[0]
        missing_count = self.df[self.selected_column].isnull().sum()
        self.missingCountLabel.setText(f"Missing values in {self.selected_column}: {missing_count}")
        self.show_column_widgets()

        if self.selected_column in self.datetime_columns:
            self.strategyDropdown.clear()
            self.strategyDropdown.addItem("Drop rows")
            self.specificValueEdit.hide()
        elif self.selected_column in self.categorical_columns:
            self.strategyDropdown.clear()
            self.strategyDropdown.addItems(["Fill with mode", "Fill with specific value", "Drop rows"])
        elif self.selected_column in self.numeric_columns:
            self.strategyDropdown.clear()
            self.strategyDropdown.addItems(["Fill with mean", "Fill with median", "Fill with mode", "Fill with specific value", "Drop rows"])

    def lag_column_clicked(self, item):
        self.selected_lag_column = item.text().split(" - ")[0]

    def rolling_column_clicked(self, item):
        self.selected_rolling_column = item.text().split(" - ")[0]

    def cumulative_column_clicked(self, item):
        self.selected_cumulative_column = item.text().split(" - ")[0]

    def feature_selection_column_clicked(self, item):
        self.selected_feature_column = item.text().split(" - ")[0]

    def delete_selected_feature_column(self):
        if self.selected_feature_column:
            self.df.drop(columns=[self.selected_feature_column], inplace=True)
            self.update_table_view()
            self.populate_column_list()

    def select_target_column(self):
        if self.selected_column:
            self.target_column = self.selected_column
            QtWidgets.QMessageBox.information(self, "Target Column Selected", f"Target column set to: {self.target_column}")
            self.populate_column_list()

    def apply_strategy(self):
        strategy = self.strategyDropdown.currentText()
        column = self.selected_column
        try:
            if strategy == "Fill with mean":
                mean_value = self.df[column].mean()
                if pd.api.types.is_integer_dtype(self.df[column]):
                    mean_value = round(mean_value)
                self.df[column].fillna(mean_value, inplace=True)
            elif strategy == "Fill with median":
                median_value = self.df[column].median()
                if pd.api.types.is_integer_dtype(self.df[column]):
                    median_value = round(median_value)
                self.df[column].fillna(median_value, inplace=True)
            elif strategy == "Fill with mode":
                mode_value = self.df[column].mode().iloc[0]
                if pd.api.types.is_integer_dtype(self.df[column]):
                    mode_value = round(mode_value)
                self.df[column].fillna(mode_value, inplace=True)
            elif strategy == "Fill with specific value":
                value = self.specificValueEdit.text()
                if value:
                    try:
                        if pd.api.types.is_numeric_dtype(self.df[column]):
                            self.df[column].fillna(float(value), inplace=True)
                        else:
                            self.df[column].fillna(value, inplace=True)
                    except ValueError:
                        QtWidgets.QMessageBox.warning(self, "Invalid Input", f"Cannot fill column {column} with value '{value}'. Please enter a valid value.")
                        return
            elif strategy == "Drop rows":
                self.df.dropna(subset=[column], inplace=True)
                self.df.reset_index(drop=True, inplace=True)
            self.update_table_view()
            self.populate_column_list()
        except Exception as e:
            self.errorLabel.setText(str(e))
            self.errorLabel.show()

    def engineer_features_changed(self):
        current_index = self.engineerFeaturesDropdown.currentIndex()
        self.optionsStackedWidget.setCurrentIndex(current_index)

    def apply_feature_engineering(self):
        self.errorLabel.hide()
        option = self.engineerFeaturesDropdown.currentText()
        if option == "Polynomial features":
            degree = int(self.degreeDropdown.currentText().split(" - ")[1])
            self.apply_polynomial_features(degree)
        elif option == "Date-time feature extraction":
            self.apply_datetime_extraction()
        elif option == "Lag features":
            self.apply_lag_features()
        elif option == "Rolling statistics":
            self.apply_rolling_statistics()
        elif option == "Cumulative statistics":
            self.apply_cumulative_statistics()

    def apply_polynomial_features(self, degree):
        poly = PolynomialFeatures(degree)
        numeric_columns = self.df.drop(columns=[self.target_column]).select_dtypes(include=[float, int]).columns

        for column in numeric_columns:
            if self.df[column].isnull().any():
                QtWidgets.QMessageBox.warning(self, "Missing Values", f"Missing values found in column {column}. Please handle missing values before applying polynomial features.")
                return

        for column in numeric_columns:
            transformed = poly.fit_transform(self.df[[column]])
            col_names = [f"{column}_poly_{i}" for i in range(transformed.shape[1])]
            poly_df = pd.DataFrame(transformed, columns=col_names)
            self.df = pd.concat([self.df, poly_df.iloc[:, 1:]], axis=1)

            self.numeric_columns.extend(col_names[1:])

        self.update_table_view()
        self.update_column_list()

    def apply_datetime_extraction(self):
        try:
            if not self.selected_datetime_column:
                raise ValueError("No datetime column selected.")
            date_format = '%d/%m/%Y'
            date_series = pd.to_datetime(self.df[self.selected_datetime_column], format=date_format, errors='coerce')
            if date_series.isnull().all():
                raise ValueError(f"The column '{self.selected_datetime_column}' cannot be converted to datetime with the given format.")
            
            if self.extractYearFromDatetime.isChecked():
                year_col = f'{self.selected_datetime_column}_year'
                self.df[year_col] = date_series.dt.year
                if year_col not in self.categorical_columns:
                    self.categorical_columns.append(year_col)
            if self.extractMonthFromDatetime.isChecked():
                month_col = f'{self.selected_datetime_column}_month'
                self.df[month_col] = date_series.dt.month
                if month_col not in self.categorical_columns:
                    self.categorical_columns.append(month_col)
            if self.extractDayFromDatetime.isChecked():
                day_col = f'{self.selected_datetime_column}_day'
                self.df[day_col] = date_series.dt.day
                if day_col not in self.categorical_columns:
                    self.categorical_columns.append(day_col)
            if self.extractDOWFromDatetime.isChecked():
                dow_col = f'{self.selected_datetime_column}_dayofweek'
                self.df[dow_col] = date_series.dt.dayofweek
                if dow_col not in self.categorical_columns:
                    self.categorical_columns.append(dow_col)

            self.update_table_view()
            self.update_column_list()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Column", str(e))

    def apply_lag_features(self):
        try:
            lag_value = int(self.lagSpaceValueEdit.text())
            column = self.selected_lag_column
            new_column_name = f'{column}_lag_{lag_value}'
            self.df[new_column_name] = self.df[column].shift(lag_value)

            if new_column_name not in self.numeric_columns:
                self.numeric_columns.append(new_column_name)

            self.update_table_view()
            self.update_column_list()
        except ValueError:
            self.errorLabel.setText("Please enter a valid integer for the lag value.")
            self.errorLabel.show()
        except AttributeError:
            self.errorLabel.setText("Please select a column for lag features.")
            self.errorLabel.show()

    def apply_rolling_statistics(self):
        try:
            window_size = int(self.rollingWindowValueEdit.text())
            column = self.selected_rolling_column
            rolling_option = self.rollingOptionDropdown.currentText()
            new_column_name = f'{column}_rolling_{rolling_option.replace(" ", "_").lower()}_{window_size}'
            if rolling_option == "Rolling sum":
                self.df[new_column_name] = self.df[column].rolling(window=window_size).sum()
            elif rolling_option == "Rolling mean":
                self.df[new_column_name] = self.df[column].rolling(window=window_size).mean()
            elif rolling_option == "Rolling standard deviation":
                self.df[new_column_name] = self.df[column].rolling(window=window_size).std()
            elif rolling_option == "Rolling minimum":
                self.df[new_column_name] = self.df[column].rolling(window=window_size).min()
            elif rolling_option == "Rolling maximum":
                self.df[new_column_name] = self.df[column].rolling(window=window_size).max()

            if new_column_name not in self.numeric_columns:
                self.numeric_columns.append(new_column_name)

            self.update_table_view()
            self.update_column_list()
        except ValueError:
            self.errorLabel.setText("Please enter a valid integer for the window size.")
            self.errorLabel.show()
        except AttributeError:
            self.errorLabel.setText("Please select a column for rolling statistics.")
            self.errorLabel.show()

    def apply_cumulative_statistics(self):
        try:
            column = self.selected_cumulative_column
            cumulative_option = self.cumulativeOptionDropdown.currentText()
            new_column_name = f'{column}_cum_{cumulative_option.replace(" ", "_").lower()}'
            if cumulative_option == "Cumulative sum":
                self.df[new_column_name] = self.df[column].cumsum()
            elif cumulative_option == "Cumulative mean":
                self.df[new_column_name] = self.df[column].expanding().mean()
            elif cumulative_option == "Cumulative minimum":
                self.df[new_column_name] = self.df[column].cummin()
            elif cumulative_option == "Cumulative maximum":
                self.df[new_column_name] = self.df[column].cummax()
            elif cumulative_option == "Cumulative product":
                self.df[new_column_name] = self.df[column].cumprod()

            if new_column_name not in self.numeric_columns:
                self.numeric_columns.append(new_column_name)

            self.update_table_view()
            self.update_column_list()
        except AttributeError:
            self.errorLabel.setText("Please select a column for cumulative statistics.")
            self.errorLabel.show()

    def apply_scaling(self):
        apply_label_encoding = self.applyCategoricalEncodingCheckbox.isChecked()
        apply_scaling = self.applyNumericalScalingCheckbox.isChecked()
        
        if apply_label_encoding:
            label_encoders = {}
            for column in self.df.columns:
                if column in self.categorical_columns and column not in self.datetime_columns:
                    le = LabelEncoder()
                    self.df[column] = le.fit_transform(self.df[column])
                    label_encoders[column] = le

        if apply_scaling:
            scaling_option = self.numericalFeatureScalingOptionDropdown.currentText()
            numeric_columns = [col for col in self.numeric_columns if col not in self.datetime_columns]
            if scaling_option == "Normalization":
                self.df[numeric_columns] = MinMaxScaler().fit_transform(self.df[numeric_columns])
            elif scaling_option == "Standardization":
                self.df[numeric_columns] = StandardScaler().fit_transform(self.df[numeric_columns])
            elif scaling_option == "Robust scaling":
                self.df[numeric_columns] = RobustScaler().fit_transform(self.df[numeric_columns])
            elif scaling_option == "MaxAbs scaling":
                self.df[numeric_columns] = MaxAbsScaler().fit_transform(self.df[numeric_columns])
        self.update_table_view()

    def feature_selection_method_changed(self):
        current_index = self.selectFeaturesOptionDropdown.currentIndex()
        self.stackedWidget.setCurrentIndex(current_index)

    def apply_feature_selection(self):
        if not self.target_column:
            QtWidgets.QMessageBox.warning(self, "No Target Column", "Please select a target column before applying feature selection.")
            return

        option = self.selectFeaturesOptionDropdown.currentText()
        if option == "Correlation coefficient":
            threshold = float(self.correlationCoefficientThresholdValueEdit.text())
            self.apply_correlation_coefficient(threshold)
        elif option == "Chi-square test":
            num_features = int(self.chiSquareTestFeaturesValueEdit.text())
            self.apply_chi_square_test(num_features)
        elif option == "Recursive feature elimination":
            estimator_name = self.recursiveFeatureEliminationOptionDropdown.currentText()
            num_features = int(self.recursiveFeatureEliminationFeaturesValueEdit.text())
            self.apply_recursive_feature_elimination(estimator_name, num_features)
        elif option == "Lasso regression":
            alpha = float(self.lassoRegressionAlphaValueEdit.text())
            self.apply_lasso_regression(alpha)

    def apply_correlation_coefficient(self, threshold):
        try:
            numeric_df = self.df.drop(columns=[self.target_column]).select_dtypes(include=[float, int])
            corr_matrix = numeric_df.corr().abs()

            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_)
            )

            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

            self.df.drop(columns=to_drop, inplace=True)

            self.numeric_columns = [col for col in self.numeric_columns if col not in to_drop]
            self.update_column_list()
            self.update_table_view()
        except Exception as e:
            self.errorLabel.setText(str(e))
            self.errorLabel.show()

    def apply_chi_square_test(self, k):
        try:
            X = self.df.drop(columns=[self.target_column])
            X = X.drop(columns=self.datetime_columns)
            y = self.df[self.target_column]
            chi_selector = SelectKBest(chi2, k=k)
            chi_selector.fit(X, y)
            selected_features = X.columns[chi_selector.get_support()]
            self.df = self.df[selected_features.tolist() + [self.target_column]]
            self.update_table_view()
            self.populate_column_list()
        except Exception as e:
            self.errorLabel.setText(str(e))
            self.errorLabel.show()

    def apply_recursive_feature_elimination(self, estimator_name, k):
        estimator = self.get_rfe_estimator(estimator_name)
        if estimator:
            self.apply_rfe(estimator, k)
        else:
            QtWidgets.QMessageBox.warning(self, "Invalid Estimator", "Please select a valid estimator for RFE.")

    def apply_rfe(self, estimator, k):
        try:
            X = self.df.drop(columns=[self.target_column])
            X = X.drop(columns=self.datetime_columns)
            y = self.df[self.target_column]
            rfe_selector = RFE(estimator, n_features_to_select=k)
            rfe_selector.fit(X, y)
            selected_features = X.columns[rfe_selector.get_support()]
            self.df = self.df[selected_features.tolist() + [self.target_column]]
            self.update_table_view()
            self.populate_column_list()
        except Exception as e:
            self.errorLabel.setText(str(e))
            self.errorLabel.show()

    def get_rfe_estimator(self, estimator_name):
        if estimator_name == "Linear regression":
            return LinearRegression()
        elif estimator_name == "Logistic regression":
            return LogisticRegression()
        elif estimator_name == "Random forest":
            return RandomForestClassifier()
        elif estimator_name == "Support vector classifier":
            return SVC(kernel="linear")

    def apply_lasso_regression(self, alpha):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        selected_features = X.columns[lasso.coef_ != 0]
        self.df = self.df[selected_features.tolist() + [self.target_column]]
        self.update_table_view()
        self.populate_column_list()

    def select_save_location(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.save_folder_path = folder_path
            self.errorLabel.setText(f"Save location: {folder_path}")
            self.errorLabel.show()

    def save_file(self):
        if not self.save_folder_path:
            QtWidgets.QMessageBox.warning(self, "No Folder Selected", "Please select a folder to save the files.")
            return
        
        if not self.settings_are_valid():
            QtWidgets.QMessageBox.warning(self, "Invalid Settings", "Please ensure all columns have a type, and valid test size and random state values are set.")
            return

        try:
            train_df, test_df = train_test_split(self.df, test_size=self.test_size, random_state=self.random_state)

            train_path = f"{self.save_folder_path}/train.csv"
            test_path = f"{self.save_folder_path}/test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            QtWidgets.QMessageBox.information(self, "Save File", f"Files saved successfully!\nTrain: {train_path}\nTest: {test_path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"An error occurred while saving the files: {str(e)}")

    def update_test_size(self):
        try:
            self.test_size = float(self.testSizeValueEdit.text())
        except ValueError:
            self.test_size = 0.2

    def update_random_state(self):
        try:
            self.random_state = int(self.randomStateValueEdit.text())
        except ValueError:
            self.random_state = 42

    def set_input_validator(self, column):
        self.specificValueEdit.clear()
        self.specificValueEdit.setValidator(None)
        self.specificValueEdit.setPlaceholderText("")
        if pd.api.types.is_numeric_dtype(self.df[column]):
            if pd.api.types.is_integer_dtype(self.df[column]):
                self.specificValueEdit.setValidator(QIntValidator())
            else:
                self.specificValueEdit.setValidator(QDoubleValidator())
        else:
            self.specificValueEdit.setValidator(None)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
