# IMS-Dependency_Parsing
This is a project in the Master's course "Statistical Dependency Parsing" at Winter Semester 22/23 at the University of Stuttgart.
This follows an Arc-Standard approach with a neural classifier.

### How to run?
`python run --train_file wsj_train.conll06 --dev_file wsj_dev.conll06.gold --test_file wsj_test.conll06.blind --save wsj_test.conll06.pred --epoch 5`
- Run the run.py with arguments of training file, dev file, test file, and the number of epochs as an argument.
