# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DataPrep(object):
    def setupUi(self, DataPrep):
        DataPrep.setObjectName("DataPrep")
        DataPrep.resize(800, 800)
        DataPrep.setMinimumSize(QtCore.QSize(800, 800))
        DataPrep.setMaximumSize(QtCore.QSize(800, 800))
        self.centralwidget = QtWidgets.QWidget(DataPrep)
        self.centralwidget.setObjectName("centralwidget")
        self.browseFiles = QtWidgets.QPushButton(self.centralwidget)
        self.browseFiles.setGeometry(QtCore.QRect(10, 10, 100, 30))
        self.browseFiles.setToolTipDuration(-5)
        self.browseFiles.setObjectName("browseFiles")
        self.rawFilePath = QtWidgets.QLineEdit(self.centralwidget)
        self.rawFilePath.setGeometry(QtCore.QRect(120, 10, 670, 30))
        self.rawFilePath.setObjectName("rawFilePath")
        self.csvFilePreview = QtWidgets.QTableView(self.centralwidget)
        self.csvFilePreview.setGeometry(QtCore.QRect(10, 50, 780, 240))
        self.csvFilePreview.setObjectName("csvFilePreview")
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(690, 760, 100, 30))
        self.saveButton.setObjectName("saveButton")
        self.navigationBar = QtWidgets.QTabWidget(self.centralwidget)
        self.navigationBar.setGeometry(QtCore.QRect(10, 300, 780, 450))
        self.navigationBar.setToolTip("")
        self.navigationBar.setObjectName("navigationBar")
        self.settings = QtWidgets.QWidget()
        self.settings.setObjectName("settings")
        self.settingsSectionDivider = QtWidgets.QFrame(self.settings)
        self.settingsSectionDivider.setGeometry(QtCore.QRect(380, 10, 20, 400))
        self.settingsSectionDivider.setFrameShape(QtWidgets.QFrame.VLine)
        self.settingsSectionDivider.setFrameShadow(QtWidgets.QFrame.Raised)
        self.settingsSectionDivider.setObjectName("settingsSectionDivider")
        self.settingsTooltipDivider = QtWidgets.QFrame(self.settings)
        self.settingsTooltipDivider.setGeometry(QtCore.QRect(400, 200, 365, 20))
        self.settingsTooltipDivider.setFrameShape(QtWidgets.QFrame.HLine)
        self.settingsTooltipDivider.setFrameShadow(QtWidgets.QFrame.Plain)
        self.settingsTooltipDivider.setObjectName("settingsTooltipDivider")
        self.chooseNCDColumnsLabel = QtWidgets.QLabel(self.settings)
        self.chooseNCDColumnsLabel.setGeometry(QtCore.QRect(10, 10, 290, 30))
        self.chooseNCDColumnsLabel.setObjectName("chooseNCDColumnsLabel")
        self.settingsColumnList = QtWidgets.QListWidget(self.settings)
        self.settingsColumnList.setGeometry(QtCore.QRect(10, 40, 370, 200))
        self.settingsColumnList.setObjectName("settingsColumnList")
        self.selectAsCategoricalButton = QtWidgets.QPushButton(self.settings)
        self.selectAsCategoricalButton.setGeometry(QtCore.QRect(105, 250, 85, 30))
        self.selectAsCategoricalButton.setObjectName("selectAsCategoricalButton")
        self.selectAsNumericalButton = QtWidgets.QPushButton(self.settings)
        self.selectAsNumericalButton.setGeometry(QtCore.QRect(10, 250, 85, 30))
        self.selectAsNumericalButton.setObjectName("selectAsNumericalButton")
        self.selectAsDatetimeButton = QtWidgets.QPushButton(self.settings)
        self.selectAsDatetimeButton.setGeometry(QtCore.QRect(200, 250, 85, 30))
        self.selectAsDatetimeButton.setObjectName("selectAsDatetimeButton")
        self.testSizeLabel = QtWidgets.QLabel(self.settings)
        self.testSizeLabel.setGeometry(QtCore.QRect(400, 10, 60, 30))
        self.testSizeLabel.setObjectName("testSizeLabel")
        self.randomStateLabel = QtWidgets.QLabel(self.settings)
        self.randomStateLabel.setGeometry(QtCore.QRect(400, 50, 120, 30))
        self.randomStateLabel.setObjectName("randomStateLabel")
        self.testSizeValueEdit = QtWidgets.QLineEdit(self.settings)
        self.testSizeValueEdit.setGeometry(QtCore.QRect(470, 10, 50, 30))
        self.testSizeValueEdit.setObjectName("testSizeValueEdit")
        self.randomStateValueEdit = QtWidgets.QLineEdit(self.settings)
        self.randomStateValueEdit.setGeometry(QtCore.QRect(530, 50, 50, 30))
        self.randomStateValueEdit.setObjectName("randomStateValueEdit")
        self.selectTargetColumnButton = QtWidgets.QPushButton(self.settings)
        self.selectTargetColumnButton.setGeometry(QtCore.QRect(295, 250, 85, 30))
        self.selectTargetColumnButton.setObjectName("selectTargetColumnButton")
        self.navigationBar.addTab(self.settings, "")
        self.handleMissingValues = QtWidgets.QWidget()
        self.handleMissingValues.setObjectName("handleMissingValues")
        self.handleMissingValuesSectionDivider = QtWidgets.QFrame(self.handleMissingValues)
        self.handleMissingValuesSectionDivider.setGeometry(QtCore.QRect(380, 10, 20, 400))
        self.handleMissingValuesSectionDivider.setFrameShape(QtWidgets.QFrame.VLine)
        self.handleMissingValuesSectionDivider.setFrameShadow(QtWidgets.QFrame.Raised)
        self.handleMissingValuesSectionDivider.setObjectName("handleMissingValuesSectionDivider")
        self.handleMissingValuesColumnList = QtWidgets.QListWidget(self.handleMissingValues)
        self.handleMissingValuesColumnList.setGeometry(QtCore.QRect(10, 10, 370, 200))
        self.handleMissingValuesColumnList.setObjectName("handleMissingValuesColumnList")
        self.missingCountLabel = QtWidgets.QLabel(self.handleMissingValues)
        self.missingCountLabel.setGeometry(QtCore.QRect(10, 220, 370, 30))
        self.missingCountLabel.setText("")
        self.missingCountLabel.setObjectName("missingCountLabel")
        self.strategyDropdown = QtWidgets.QComboBox(self.handleMissingValues)
        self.strategyDropdown.setGeometry(QtCore.QRect(10, 260, 180, 30))
        self.strategyDropdown.setObjectName("strategyDropdown")
        self.strategyDropdown.addItem("")
        self.strategyDropdown.addItem("")
        self.strategyDropdown.addItem("")
        self.strategyDropdown.addItem("")
        self.strategyDropdown.addItem("")
        self.specificValueEdit = QtWidgets.QLineEdit(self.handleMissingValues)
        self.specificValueEdit.setGeometry(QtCore.QRect(200, 260, 180, 30))
        self.specificValueEdit.setObjectName("specificValueEdit")
        self.handleMissingValuesApplyButton = QtWidgets.QPushButton(self.handleMissingValues)
        self.handleMissingValuesApplyButton.setGeometry(QtCore.QRect(280, 380, 100, 30))
        self.handleMissingValuesApplyButton.setObjectName("handleMissingValuesApplyButton")
        self.handleMissingValuesTooltipDivider = QtWidgets.QFrame(self.handleMissingValues)
        self.handleMissingValuesTooltipDivider.setGeometry(QtCore.QRect(400, 200, 365, 20))
        self.handleMissingValuesTooltipDivider.setFrameShape(QtWidgets.QFrame.HLine)
        self.handleMissingValuesTooltipDivider.setFrameShadow(QtWidgets.QFrame.Plain)
        self.handleMissingValuesTooltipDivider.setObjectName("handleMissingValuesTooltipDivider")
        self.handleMissingValuesExplanationLabel = QtWidgets.QLabel(self.handleMissingValues)
        self.handleMissingValuesExplanationLabel.setEnabled(True)
        self.handleMissingValuesExplanationLabel.setGeometry(QtCore.QRect(400, 10, 365, 190))
        self.handleMissingValuesExplanationLabel.setTextFormat(QtCore.Qt.RichText)
        self.handleMissingValuesExplanationLabel.setWordWrap(True)
        self.handleMissingValuesExplanationLabel.setObjectName("handleMissingValuesExplanationLabel")
        self.navigationBar.addTab(self.handleMissingValues, "")
        self.engineerFeatures = QtWidgets.QWidget()
        self.engineerFeatures.setObjectName("engineerFeatures")
        self.engineerFeaturesSectionDivider = QtWidgets.QFrame(self.engineerFeatures)
        self.engineerFeaturesSectionDivider.setGeometry(QtCore.QRect(380, 10, 20, 400))
        self.engineerFeaturesSectionDivider.setFrameShape(QtWidgets.QFrame.VLine)
        self.engineerFeaturesSectionDivider.setFrameShadow(QtWidgets.QFrame.Raised)
        self.engineerFeaturesSectionDivider.setObjectName("engineerFeaturesSectionDivider")
        self.engineerFeaturesDropdown = QtWidgets.QComboBox(self.engineerFeatures)
        self.engineerFeaturesDropdown.setGeometry(QtCore.QRect(10, 10, 180, 30))
        self.engineerFeaturesDropdown.setObjectName("engineerFeaturesDropdown")
        self.engineerFeaturesDropdown.addItem("")
        self.engineerFeaturesDropdown.addItem("")
        self.engineerFeaturesDropdown.addItem("")
        self.engineerFeaturesDropdown.addItem("")
        self.engineerFeaturesDropdown.addItem("")
        self.optionsStackedWidget = QtWidgets.QStackedWidget(self.engineerFeatures)
        self.optionsStackedWidget.setGeometry(QtCore.QRect(10, 50, 370, 320))
        self.optionsStackedWidget.setObjectName("optionsStackedWidget")
        self.polynomialDegree = QtWidgets.QWidget()
        self.polynomialDegree.setObjectName("polynomialDegree")
        self.degreeDropdown = QtWidgets.QComboBox(self.polynomialDegree)
        self.degreeDropdown.setGeometry(QtCore.QRect(0, 0, 180, 30))
        self.degreeDropdown.setObjectName("degreeDropdown")
        self.degreeDropdown.addItem("")
        self.degreeDropdown.addItem("")
        self.degreeDropdown.addItem("")
        self.optionsStackedWidget.addWidget(self.polynomialDegree)
        self.dateTimeExtraction = QtWidgets.QWidget()
        self.dateTimeExtraction.setObjectName("dateTimeExtraction")
        self.extractYearFromDatetime = QtWidgets.QCheckBox(self.dateTimeExtraction)
        self.extractYearFromDatetime.setGeometry(QtCore.QRect(0, 0, 370, 30))
        self.extractYearFromDatetime.setObjectName("extractYearFromDatetime")
        self.extractMonthFromDatetime = QtWidgets.QCheckBox(self.dateTimeExtraction)
        self.extractMonthFromDatetime.setGeometry(QtCore.QRect(0, 30, 370, 30))
        self.extractMonthFromDatetime.setObjectName("extractMonthFromDatetime")
        self.extractDayFromDatetime = QtWidgets.QCheckBox(self.dateTimeExtraction)
        self.extractDayFromDatetime.setGeometry(QtCore.QRect(0, 60, 370, 30))
        self.extractDayFromDatetime.setObjectName("extractDayFromDatetime")
        self.extractDOWFromDatetime = QtWidgets.QCheckBox(self.dateTimeExtraction)
        self.extractDOWFromDatetime.setGeometry(QtCore.QRect(0, 90, 370, 30))
        self.extractDOWFromDatetime.setObjectName("extractDOWFromDatetime")
        self.optionsStackedWidget.addWidget(self.dateTimeExtraction)
        self.lagFeatures = QtWidgets.QWidget()
        self.lagFeatures.setObjectName("lagFeatures")
        self.lagFeaturesEngineerFeaturesColumnList = QtWidgets.QListWidget(self.lagFeatures)
        self.lagFeaturesEngineerFeaturesColumnList.setGeometry(QtCore.QRect(0, 0, 370, 200))
        self.lagFeaturesEngineerFeaturesColumnList.setObjectName("lagFeaturesEngineerFeaturesColumnList")
        self.chooseLagSpaceLabel = QtWidgets.QLabel(self.lagFeatures)
        self.chooseLagSpaceLabel.setGeometry(QtCore.QRect(0, 210, 250, 30))
        self.chooseLagSpaceLabel.setObjectName("chooseLagSpaceLabel")
        self.lagSpaceValueEdit = QtWidgets.QLineEdit(self.lagFeatures)
        self.lagSpaceValueEdit.setGeometry(QtCore.QRect(250, 210, 120, 30))
        self.lagSpaceValueEdit.setObjectName("lagSpaceValueEdit")
        self.optionsStackedWidget.addWidget(self.lagFeatures)
        self.rollingStatistics = QtWidgets.QWidget()
        self.rollingStatistics.setObjectName("rollingStatistics")
        self.rollingStatisticsEngineerFeaturesColumnList = QtWidgets.QListWidget(self.rollingStatistics)
        self.rollingStatisticsEngineerFeaturesColumnList.setGeometry(QtCore.QRect(0, 0, 370, 200))
        self.rollingStatisticsEngineerFeaturesColumnList.setObjectName("rollingStatisticsEngineerFeaturesColumnList")
        self.rollingOptionDropdown = QtWidgets.QComboBox(self.rollingStatistics)
        self.rollingOptionDropdown.setGeometry(QtCore.QRect(0, 210, 180, 30))
        self.rollingOptionDropdown.setObjectName("rollingOptionDropdown")
        self.rollingOptionDropdown.addItem("")
        self.rollingOptionDropdown.addItem("")
        self.rollingOptionDropdown.addItem("")
        self.rollingOptionDropdown.addItem("")
        self.rollingOptionDropdown.addItem("")
        self.rollingStatisticsWindowLabel = QtWidgets.QLabel(self.rollingStatistics)
        self.rollingStatisticsWindowLabel.setGeometry(QtCore.QRect(190, 210, 55, 30))
        self.rollingStatisticsWindowLabel.setObjectName("rollingStatisticsWindowLabel")
        self.rollingWindowValueEdit = QtWidgets.QLineEdit(self.rollingStatistics)
        self.rollingWindowValueEdit.setGeometry(QtCore.QRect(250, 210, 120, 30))
        self.rollingWindowValueEdit.setObjectName("rollingWindowValueEdit")
        self.optionsStackedWidget.addWidget(self.rollingStatistics)
        self.cumulativeStatistics = QtWidgets.QWidget()
        self.cumulativeStatistics.setObjectName("cumulativeStatistics")
        self.cumulativeStatisticsEngineerFeaturesColumnList = QtWidgets.QListWidget(self.cumulativeStatistics)
        self.cumulativeStatisticsEngineerFeaturesColumnList.setGeometry(QtCore.QRect(0, 0, 370, 200))
        self.cumulativeStatisticsEngineerFeaturesColumnList.setObjectName("cumulativeStatisticsEngineerFeaturesColumnList")
        self.cumulativeOptionDropdown = QtWidgets.QComboBox(self.cumulativeStatistics)
        self.cumulativeOptionDropdown.setGeometry(QtCore.QRect(0, 210, 180, 30))
        self.cumulativeOptionDropdown.setObjectName("cumulativeOptionDropdown")
        self.cumulativeOptionDropdown.addItem("")
        self.cumulativeOptionDropdown.addItem("")
        self.cumulativeOptionDropdown.addItem("")
        self.cumulativeOptionDropdown.addItem("")
        self.cumulativeOptionDropdown.addItem("")
        self.optionsStackedWidget.addWidget(self.cumulativeStatistics)
        self.engineerFeaturesApplyButton = QtWidgets.QPushButton(self.engineerFeatures)
        self.engineerFeaturesApplyButton.setGeometry(QtCore.QRect(280, 380, 100, 30))
        self.engineerFeaturesApplyButton.setObjectName("engineerFeaturesApplyButton")
        self.engineerFeaturesTooltipDivider = QtWidgets.QFrame(self.engineerFeatures)
        self.engineerFeaturesTooltipDivider.setGeometry(QtCore.QRect(400, 200, 365, 20))
        self.engineerFeaturesTooltipDivider.setFrameShape(QtWidgets.QFrame.HLine)
        self.engineerFeaturesTooltipDivider.setFrameShadow(QtWidgets.QFrame.Plain)
        self.engineerFeaturesTooltipDivider.setObjectName("engineerFeaturesTooltipDivider")
        self.engineerFeaturesExplanationLabel = QtWidgets.QLabel(self.engineerFeatures)
        self.engineerFeaturesExplanationLabel.setGeometry(QtCore.QRect(400, 10, 365, 190))
        self.engineerFeaturesExplanationLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.engineerFeaturesExplanationLabel.setWordWrap(True)
        self.engineerFeaturesExplanationLabel.setObjectName("engineerFeaturesExplanationLabel")
        self.navigationBar.addTab(self.engineerFeatures, "")
        self.scale = QtWidgets.QWidget()
        self.scale.setObjectName("scale")
        self.scaleSectionDivider = QtWidgets.QFrame(self.scale)
        self.scaleSectionDivider.setGeometry(QtCore.QRect(380, 10, 20, 400))
        self.scaleSectionDivider.setFrameShape(QtWidgets.QFrame.VLine)
        self.scaleSectionDivider.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scaleSectionDivider.setObjectName("scaleSectionDivider")
        self.numericalFeatureScalingOptionDropdown = QtWidgets.QComboBox(self.scale)
        self.numericalFeatureScalingOptionDropdown.setGeometry(QtCore.QRect(10, 90, 150, 30))
        self.numericalFeatureScalingOptionDropdown.setObjectName("numericalFeatureScalingOptionDropdown")
        self.numericalFeatureScalingOptionDropdown.addItem("")
        self.numericalFeatureScalingOptionDropdown.addItem("")
        self.numericalFeatureScalingOptionDropdown.addItem("")
        self.numericalFeatureScalingOptionDropdown.addItem("")
        self.applyCategoricalEncodingCheckbox = QtWidgets.QCheckBox(self.scale)
        self.applyCategoricalEncodingCheckbox.setGeometry(QtCore.QRect(10, 10, 220, 30))
        self.applyCategoricalEncodingCheckbox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.applyCategoricalEncodingCheckbox.setAutoFillBackground(False)
        self.applyCategoricalEncodingCheckbox.setObjectName("applyCategoricalEncodingCheckbox")
        self.applyScalingButton = QtWidgets.QPushButton(self.scale)
        self.applyScalingButton.setGeometry(QtCore.QRect(280, 380, 100, 30))
        self.applyScalingButton.setObjectName("applyScalingButton")
        self.applyNumericalScalingCheckbox = QtWidgets.QCheckBox(self.scale)
        self.applyNumericalScalingCheckbox.setGeometry(QtCore.QRect(10, 50, 220, 30))
        self.applyNumericalScalingCheckbox.setObjectName("applyNumericalScalingCheckbox")
        self.scaleTooltipDivider = QtWidgets.QFrame(self.scale)
        self.scaleTooltipDivider.setGeometry(QtCore.QRect(400, 200, 365, 20))
        self.scaleTooltipDivider.setFrameShape(QtWidgets.QFrame.HLine)
        self.scaleTooltipDivider.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scaleTooltipDivider.setObjectName("scaleTooltipDivider")
        self.scaleExplanationLabel = QtWidgets.QLabel(self.scale)
        self.scaleExplanationLabel.setGeometry(QtCore.QRect(400, 10, 365, 190))
        self.scaleExplanationLabel.setWordWrap(True)
        self.scaleExplanationLabel.setObjectName("scaleExplanationLabel")
        self.navigationBar.addTab(self.scale, "")
        self.selectFeatures = QtWidgets.QWidget()
        self.selectFeatures.setObjectName("selectFeatures")
        self.selectFeaturesSectionDivider = QtWidgets.QFrame(self.selectFeatures)
        self.selectFeaturesSectionDivider.setGeometry(QtCore.QRect(380, 10, 20, 400))
        self.selectFeaturesSectionDivider.setFrameShape(QtWidgets.QFrame.VLine)
        self.selectFeaturesSectionDivider.setFrameShadow(QtWidgets.QFrame.Raised)
        self.selectFeaturesSectionDivider.setObjectName("selectFeaturesSectionDivider")
        self.selectFeaturesTooltipDivider = QtWidgets.QFrame(self.selectFeatures)
        self.selectFeaturesTooltipDivider.setGeometry(QtCore.QRect(400, 200, 365, 20))
        self.selectFeaturesTooltipDivider.setFrameShape(QtWidgets.QFrame.HLine)
        self.selectFeaturesTooltipDivider.setFrameShadow(QtWidgets.QFrame.Plain)
        self.selectFeaturesTooltipDivider.setObjectName("selectFeaturesTooltipDivider")
        self.selectFeaturesExplanationLabel = QtWidgets.QLabel(self.selectFeatures)
        self.selectFeaturesExplanationLabel.setGeometry(QtCore.QRect(400, 10, 365, 190))
        self.selectFeaturesExplanationLabel.setWordWrap(True)
        self.selectFeaturesExplanationLabel.setObjectName("selectFeaturesExplanationLabel")
        self.selectFeaturesColumnList = QtWidgets.QListWidget(self.selectFeatures)
        self.selectFeaturesColumnList.setGeometry(QtCore.QRect(10, 10, 370, 200))
        self.selectFeaturesColumnList.setObjectName("selectFeaturesColumnList")
        self.selectFeaturesDeleteColumnButton = QtWidgets.QPushButton(self.selectFeatures)
        self.selectFeaturesDeleteColumnButton.setGeometry(QtCore.QRect(295, 220, 85, 30))
        self.selectFeaturesDeleteColumnButton.setObjectName("selectFeaturesDeleteColumnButton")
        self.selectFeaturesOptionDropdown = QtWidgets.QComboBox(self.selectFeatures)
        self.selectFeaturesOptionDropdown.setGeometry(QtCore.QRect(10, 220, 180, 30))
        self.selectFeaturesOptionDropdown.setObjectName("selectFeaturesOptionDropdown")
        self.selectFeaturesOptionDropdown.addItem("")
        self.selectFeaturesOptionDropdown.addItem("")
        self.selectFeaturesOptionDropdown.addItem("")
        self.selectFeaturesOptionDropdown.addItem("")
        self.stackedWidget = QtWidgets.QStackedWidget(self.selectFeatures)
        self.stackedWidget.setGeometry(QtCore.QRect(10, 260, 370, 110))
        self.stackedWidget.setObjectName("stackedWidget")
        self.correlationCoeficient = QtWidgets.QWidget()
        self.correlationCoeficient.setObjectName("correlationCoeficient")
        self.correlationCoefficientThresholdLabel = QtWidgets.QLabel(self.correlationCoeficient)
        self.correlationCoefficientThresholdLabel.setGeometry(QtCore.QRect(0, 0, 65, 30))
        self.correlationCoefficientThresholdLabel.setObjectName("correlationCoefficientThresholdLabel")
        self.correlationCoefficientThresholdValueEdit = QtWidgets.QLineEdit(self.correlationCoeficient)
        self.correlationCoefficientThresholdValueEdit.setGeometry(QtCore.QRect(70, 0, 110, 30))
        self.correlationCoefficientThresholdValueEdit.setObjectName("correlationCoefficientThresholdValueEdit")
        self.stackedWidget.addWidget(self.correlationCoeficient)
        self.chiSquareTest = QtWidgets.QWidget()
        self.chiSquareTest.setObjectName("chiSquareTest")
        self.chiSquareTestFeaturesLabel = QtWidgets.QLabel(self.chiSquareTest)
        self.chiSquareTestFeaturesLabel.setGeometry(QtCore.QRect(0, 0, 60, 30))
        self.chiSquareTestFeaturesLabel.setObjectName("chiSquareTestFeaturesLabel")
        self.chiSquareTestFeaturesValueEdit = QtWidgets.QLineEdit(self.chiSquareTest)
        self.chiSquareTestFeaturesValueEdit.setGeometry(QtCore.QRect(70, 0, 110, 30))
        self.chiSquareTestFeaturesValueEdit.setObjectName("chiSquareTestFeaturesValueEdit")
        self.stackedWidget.addWidget(self.chiSquareTest)
        self.recursiveFeatureElimination = QtWidgets.QWidget()
        self.recursiveFeatureElimination.setObjectName("recursiveFeatureElimination")
        self.recursiveFeatureEliminationOptionDropdown = QtWidgets.QComboBox(self.recursiveFeatureElimination)
        self.recursiveFeatureEliminationOptionDropdown.setGeometry(QtCore.QRect(0, 0, 180, 30))
        self.recursiveFeatureEliminationOptionDropdown.setObjectName("recursiveFeatureEliminationOptionDropdown")
        self.recursiveFeatureEliminationOptionDropdown.addItem("")
        self.recursiveFeatureEliminationOptionDropdown.addItem("")
        self.recursiveFeatureEliminationOptionDropdown.addItem("")
        self.recursiveFeatureEliminationOptionDropdown.addItem("")
        self.recursiveFeatureEliminationFeaturesLabel = QtWidgets.QLabel(self.recursiveFeatureElimination)
        self.recursiveFeatureEliminationFeaturesLabel.setGeometry(QtCore.QRect(0, 40, 60, 30))
        self.recursiveFeatureEliminationFeaturesLabel.setObjectName("recursiveFeatureEliminationFeaturesLabel")
        self.recursiveFeatureEliminationFeaturesValueEdit = QtWidgets.QLineEdit(self.recursiveFeatureElimination)
        self.recursiveFeatureEliminationFeaturesValueEdit.setGeometry(QtCore.QRect(70, 40, 110, 30))
        self.recursiveFeatureEliminationFeaturesValueEdit.setObjectName("recursiveFeatureEliminationFeaturesValueEdit")
        self.stackedWidget.addWidget(self.recursiveFeatureElimination)
        self.lassoRegression = QtWidgets.QWidget()
        self.lassoRegression.setObjectName("lassoRegression")
        self.lassoRegressionAlphaLabel = QtWidgets.QLabel(self.lassoRegression)
        self.lassoRegressionAlphaLabel.setGeometry(QtCore.QRect(0, 0, 41, 30))
        self.lassoRegressionAlphaLabel.setObjectName("lassoRegressionAlphaLabel")
        self.lassoRegressionAlphaValueEdit = QtWidgets.QLineEdit(self.lassoRegression)
        self.lassoRegressionAlphaValueEdit.setGeometry(QtCore.QRect(50, 0, 130, 30))
        self.lassoRegressionAlphaValueEdit.setObjectName("lassoRegressionAlphaValueEdit")
        self.stackedWidget.addWidget(self.lassoRegression)
        self.applySelectFeaturesButton = QtWidgets.QPushButton(self.selectFeatures)
        self.applySelectFeaturesButton.setGeometry(QtCore.QRect(280, 380, 100, 30))
        self.applySelectFeaturesButton.setObjectName("applySelectFeaturesButton")
        self.navigationBar.addTab(self.selectFeatures, "")
        self.chooseSaveLocation = QtWidgets.QPushButton(self.centralwidget)
        self.chooseSaveLocation.setGeometry(QtCore.QRect(530, 760, 150, 30))
        self.chooseSaveLocation.setObjectName("chooseSaveLocation")
        self.readMeLabel = QtWidgets.QLabel(self.centralwidget)
        self.readMeLabel.setGeometry(QtCore.QRect(9, 49, 781, 241))
        self.readMeLabel.setObjectName("readMeLabel")
        self.errorLabel = QtWidgets.QLabel(self.centralwidget)
        self.errorLabel.setGeometry(QtCore.QRect(10, 760, 510, 30))
        self.errorLabel.setText("")
        self.errorLabel.setObjectName("errorLabel")
        DataPrep.setCentralWidget(self.centralwidget)

        self.retranslateUi(DataPrep)
        self.navigationBar.setCurrentIndex(0)
        self.optionsStackedWidget.setCurrentIndex(0)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(DataPrep)

    def retranslateUi(self, DataPrep):
        _translate = QtCore.QCoreApplication.translate
        DataPrep.setWindowTitle(_translate("DataPrep", "DataPrep"))
        self.browseFiles.setToolTip(_translate("DataPrep", "Browse local CSV files"))
        self.browseFiles.setText(_translate("DataPrep", "Browse"))
        self.saveButton.setToolTip(_translate("DataPrep", "Save to a local file"))
        self.saveButton.setText(_translate("DataPrep", "Save"))
        self.chooseNCDColumnsLabel.setText(_translate("DataPrep", "Choose numerical/categorical/date-time columns:"))
        self.selectAsCategoricalButton.setText(_translate("DataPrep", "Categorical"))
        self.selectAsNumericalButton.setText(_translate("DataPrep", "Numerical"))
        self.selectAsDatetimeButton.setText(_translate("DataPrep", "Date-time"))
        self.testSizeLabel.setText(_translate("DataPrep", "Test size:"))
        self.randomStateLabel.setText(_translate("DataPrep", "Random state value:"))
        self.selectTargetColumnButton.setText(_translate("DataPrep", "Target"))
        self.navigationBar.setTabText(self.navigationBar.indexOf(self.settings), _translate("DataPrep", "Settings"))
        self.strategyDropdown.setItemText(0, _translate("DataPrep", "Fill with mean"))
        self.strategyDropdown.setItemText(1, _translate("DataPrep", "Fill with median"))
        self.strategyDropdown.setItemText(2, _translate("DataPrep", "Fill with mode"))
        self.strategyDropdown.setItemText(3, _translate("DataPrep", "Fill with specific value"))
        self.strategyDropdown.setItemText(4, _translate("DataPrep", "Drop rows"))
        self.handleMissingValuesApplyButton.setText(_translate("DataPrep", "Apply"))
        self.handleMissingValuesExplanationLabel.setText(_translate("DataPrep", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">What it is:</span></p><p align=\"center\">Handling missing values involves identifying and addressing gaps in your dataset where data is missing. </p><p align=\"center\"><span style=\" font-weight:600;\">Why it\'s necessary:</span></p><p align=\"center\">Missing values can lead to inaccurate predictions or models that fail to generalize well. Techniques like imputation (filling in missing values) or removing incomplete records ensure the dataset is complete and reliable for model training.</p></body></html>"))
        self.navigationBar.setTabText(self.navigationBar.indexOf(self.handleMissingValues), _translate("DataPrep", "Handle missing values"))
        self.engineerFeaturesDropdown.setItemText(0, _translate("DataPrep", "Polynomial features"))
        self.engineerFeaturesDropdown.setItemText(1, _translate("DataPrep", "Date-time feature extraction"))
        self.engineerFeaturesDropdown.setItemText(2, _translate("DataPrep", "Lag features"))
        self.engineerFeaturesDropdown.setItemText(3, _translate("DataPrep", "Rolling statistics"))
        self.engineerFeaturesDropdown.setItemText(4, _translate("DataPrep", "Cumulative statistics"))
        self.degreeDropdown.setItemText(0, _translate("DataPrep", "Polynomial degree - 1"))
        self.degreeDropdown.setItemText(1, _translate("DataPrep", "Polynomial degree - 2"))
        self.degreeDropdown.setItemText(2, _translate("DataPrep", "Polynomial degree - 3"))
        self.extractYearFromDatetime.setText(_translate("DataPrep", "Extract year from date-time"))
        self.extractMonthFromDatetime.setText(_translate("DataPrep", "Extract month from date-time"))
        self.extractDayFromDatetime.setText(_translate("DataPrep", "Extract day from date-time"))
        self.extractDOWFromDatetime.setText(_translate("DataPrep", "Extract day of the week from date-time"))
        self.chooseLagSpaceLabel.setText(_translate("DataPrep", "Choose lag space fot the specified column:"))
        self.rollingOptionDropdown.setItemText(0, _translate("DataPrep", "Rolling sum"))
        self.rollingOptionDropdown.setItemText(1, _translate("DataPrep", "Rolling mean"))
        self.rollingOptionDropdown.setItemText(2, _translate("DataPrep", "Rolling standard deviation"))
        self.rollingOptionDropdown.setItemText(3, _translate("DataPrep", "Rolling mininum"))
        self.rollingOptionDropdown.setItemText(4, _translate("DataPrep", "Rolling maximum"))
        self.rollingStatisticsWindowLabel.setText(_translate("DataPrep", "Window:"))
        self.cumulativeOptionDropdown.setItemText(0, _translate("DataPrep", "Cumulative sum"))
        self.cumulativeOptionDropdown.setItemText(1, _translate("DataPrep", "Cumulative mean"))
        self.cumulativeOptionDropdown.setItemText(2, _translate("DataPrep", "Cumulative minimum"))
        self.cumulativeOptionDropdown.setItemText(3, _translate("DataPrep", "Cumulative maximum"))
        self.cumulativeOptionDropdown.setItemText(4, _translate("DataPrep", "Cumulative product"))
        self.engineerFeaturesApplyButton.setText(_translate("DataPrep", "Apply"))
        self.engineerFeaturesExplanationLabel.setText(_translate("DataPrep", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">What it is:</span></p><p align=\"center\">Feature engineering is the process of creating new features or modifying existing ones to improve the performance of a machine learning model.</p><p align=\"center\"><span style=\" font-weight:600;\">Why it\'s necessary:</span></p><p align=\"center\">Well-designed features can provide more relevant information to the model, enhancing its ability to learn patterns and make accurate predictions. This step transforms raw data into meaningful inputs.</p></body></html>"))
        self.navigationBar.setTabText(self.navigationBar.indexOf(self.engineerFeatures), _translate("DataPrep", "Engineer features"))
        self.numericalFeatureScalingOptionDropdown.setItemText(0, _translate("DataPrep", "Normalization"))
        self.numericalFeatureScalingOptionDropdown.setItemText(1, _translate("DataPrep", "Standardization"))
        self.numericalFeatureScalingOptionDropdown.setItemText(2, _translate("DataPrep", "Robust scaling"))
        self.numericalFeatureScalingOptionDropdown.setItemText(3, _translate("DataPrep", "MaxAbs scaling"))
        self.applyCategoricalEncodingCheckbox.setText(_translate("DataPrep", "Apply categorical feature encoding"))
        self.applyScalingButton.setText(_translate("DataPrep", "Apply"))
        self.applyNumericalScalingCheckbox.setText(_translate("DataPrep", "Apply numerical feature scaling"))
        self.scaleExplanationLabel.setText(_translate("DataPrep", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">What it is:</span></p><p align=\"center\">Scaling adjusts the range of features so they fit within a similar scale, typically through normalization or standardization.</p><p align=\"center\"><span style=\" font-weight:600;\">Why it\'s necessary:</span></p><p align=\"center\">Many machine learning algorithms perform better when input features are on a similar scale. This prevents certain features from dominating others and helps algorithms converge faster and more reliably.</p></body></html>"))
        self.navigationBar.setTabText(self.navigationBar.indexOf(self.scale), _translate("DataPrep", "Scale"))
        self.selectFeaturesExplanationLabel.setText(_translate("DataPrep", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">What it is:</span></p><p align=\"center\">Feature selection involves choosing the most relevant features from your dataset for training the model.</p><p align=\"center\"><span style=\" font-weight:600;\">Why it\'s necessary:</span></p><p align=\"center\">Reducing the number of features can improve model performance by eliminating noise, reducing overfitting, and decreasing computation time. It ensures the model trains on the most informative data.</p></body></html>"))
        self.selectFeaturesDeleteColumnButton.setText(_translate("DataPrep", "Delete"))
        self.selectFeaturesOptionDropdown.setItemText(0, _translate("DataPrep", "Correlation coefficient"))
        self.selectFeaturesOptionDropdown.setItemText(1, _translate("DataPrep", "Chi-square test"))
        self.selectFeaturesOptionDropdown.setItemText(2, _translate("DataPrep", "Recursive feature elimination"))
        self.selectFeaturesOptionDropdown.setItemText(3, _translate("DataPrep", "Lasso regression"))
        self.correlationCoefficientThresholdLabel.setText(_translate("DataPrep", "Threshold:"))
        self.chiSquareTestFeaturesLabel.setText(_translate("DataPrep", "Features:"))
        self.recursiveFeatureEliminationOptionDropdown.setItemText(0, _translate("DataPrep", "Linear regression"))
        self.recursiveFeatureEliminationOptionDropdown.setItemText(1, _translate("DataPrep", "Logistic regression"))
        self.recursiveFeatureEliminationOptionDropdown.setItemText(2, _translate("DataPrep", "Random forest"))
        self.recursiveFeatureEliminationOptionDropdown.setItemText(3, _translate("DataPrep", "Support vector classifier"))
        self.recursiveFeatureEliminationFeaturesLabel.setText(_translate("DataPrep", "Features:"))
        self.lassoRegressionAlphaLabel.setText(_translate("DataPrep", "Alpha:"))
        self.applySelectFeaturesButton.setText(_translate("DataPrep", "Apply"))
        self.navigationBar.setTabText(self.navigationBar.indexOf(self.selectFeatures), _translate("DataPrep", "Select features"))
        self.chooseSaveLocation.setText(_translate("DataPrep", "Choose save location"))
        self.readMeLabel.setText(_translate("DataPrep", "<html><head/><body><p align=\"center\"><span style=\"font-weight:600;\">Read me! ;)<br/>Welcome to DataPrep.<br/>Important instructions:</span></p><p align=\"center\"><span style=\"font-weight:600;\">Fill Out Settings Tab First:</span> Complete the settings tab before any preprocessing steps. This ensures the tool works properly.</p><p align=\"center\"><span style=\"font-weight:600;\">Recommended Order of Steps:</span> Follow the preprocessing steps in the interface order to maintain data integrity.</p><p align=\"center\"><span style=\"font-weight:600;\">Handle Missing Values First:</span> Address missing values before other preprocessing steps to avoid inaccuracies.</p><p align=\"center\"><span style=\"font-weight:600;\">Potential Interference:</span> Some functions may interfere with each other. Follow the recommended order to avoid issues.</p><p align=\"center\"><span style=\"font-weight:600;\">To begin, browse for CSV files.</span></p><p align=\"center\"><span style=\"font-weight:600;\">Happy preprocessing!</span></p></body></html>\n"
""))
