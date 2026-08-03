Progress log for Project Granary

2026-7-27:

Beginning work on the machine learning model (testing initial scripts)
Considering initially importing machine learning algorithms from scikit to draft the initial proof of concept
Started the scikit proof of concept model (Note: model seems to be overfitting, may need to implement new input data)
Set up new weather station console
Installed new weather station on temporary framework (Note: on old weather station base, was a temporary solution to adjust for permanent settlement later; framework was initially incompatible with new station but temporary tools were put in to address structural issues)

2026-7-28:

Connected weather station transmitter data to weatherseed app and Weather Underground app
Connected WU data to Project Granary main .csv file via Python script (Note: weather station can't relay pressure data, would have to use pressure data from nearby station)

2026-7-29:

Began setting up permanent weather station site with 4x4, tall support structure (to ensure optimal data gathering by reducing obstructions for wind and retaining structural integrity)

2026-7-30:

Began setting up the permanent weather station base (including 4x4 wooden base and concrete floor to secure the station)

2026-7-31:

Successfully integrated weather station with the base

2026-8-1 & 8-2:

Working on XGBoost multi:softmax model to construct a basic forecast system
Setting up the training data to include arrays for past 5 hours as input and an array representing the next 5 hours for output

2026-8-3:
Transitioned to VAR regression-based model after operational conflicts with the XGBoost model
