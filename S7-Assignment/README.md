Attempt 1

Targets: Define a skeleton, Make a model and use BatchNorm
Results:


Parameters: 14,536

Train Acc: 99.88%

Test Acc: 99.21%

Analysis:

The model is overfitting as train accuracy is almost close to 100% so no room for improvement
Need a smaller model with bit of regularisation


Attempt 2

Targets: Add regularization add dropout, use smaller model with GAP 
Results:

Parameters: 7,888

Train Acc: 99.24%

Test Acc: 99.28%

Analysis:

Using dropout helped with the overfitting , as we are adding dropout after each layer only a small amount 0.05 was sufficient, 0.1 reduced the test accuracy
GAP layer helped with the slow increase in layer size

Moving final layer after GAP helped with the capacity.
The test accuracy is flickering a bit, moving Max pool after RF = 5 will help


Attempt 3

Targets: Make model training more robust by using augmented data, and moving Max pool after RF=5

Results:

Parameters: 7,888

Train Acc: 99.29%

Test Acc: 99.46%

Analysis:

Getting better test accuracy after doing above targets

