Use the pre-training model: call get_pretrained_i2v directly
--------------------------------------------------------------------

Use the pre-training model provided by EduNLP to convert the given question text into vectors.

* Advantages: Simple and convenient.

* Disadvantages: Only the model given in the project can be used, which has great limitations.

* Call this function to obtain the corresponding pre-training model. At present, the following pre training models are provided: d2v_all_300, d2v_science_300, d2v_english_300 and d2v_literal_300.

Selection and use of models
####################################

Select the pre-training model according to the subject:

+--------------------+------------------------+
|   Pre-training model name  | Subject of model training data |
+====================+========================+
|    d2v_all_300     |        all subject          |
+--------------------+------------------------+
|    d2v_science_300     |         Science           |
+--------------------+------------------------+
|    d2v_literal_300     |         Arts           |
+--------------------+------------------------+
|    d2v_english_300     |         English           |
+--------------------+------------------------+

The concrete process of processing
####################################

1.Download the corresponding preprocessing model

2.Transfer the obtained model to D2V and process it with D2V
  Convert the obtained model into D2V and process it through D2V

Examples：

::

  >>> i2v = get_pretrained_i2v("d2v_science_300")
  >>> i2v(item)
