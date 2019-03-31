import tensorflow as tf
import numpy
import csv

input_file = 'processed_data_2018.csv'
PERCENT_TRAINING = 0.8
BATCH_SIZE = 14
STEPS = 4000
LEARNING_RATE = 0.001



def neural_net(features, labels, mode):

	print("Starting NN")

	dense_layer_1 = tf.layers.dense(inputs=features, units=20, activation=sigmoid)
	dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=20, activation=sigmoid)
	logits = tf.layers.dense(inputs=dense_layer_2, units=17)


	# create a map of predictions for PREDICT and EVAL modes
	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	# return the prediction if we're in the prediction mode
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# otherwise, calculate loss and train some schtuff
	loss = tf.losses.sparse_softmax_cross_entropy(
		labels=labels,
		logits=logits)


	# training
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# evaluating
	else:
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels, predictions=predictions["classes"])
		}

		return tf.estimator.EstimatorSpec(
			mode=mode, 
			loss=loss, 
			eval_metric_ops=eval_metric_ops)




def train(train_data, train_labels, classifier, iterations=50):
	
	# log information
	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=iterations)

	# train our model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=BATCH_SIZE,
		num_epochs=None,
		shuffle=True)
	classifier.train(
		input_fn=train_input_fn,
		steps=STEPS,
		hooks=[logging_hook])

	return classifier

def test(eval_data, eval_labels, classifier):
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    	x={"x": eval_data},
    	y=eval_labels,
    	num_epochs=1,
    	shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	
	return eval_results


def main(input_file):
	inputs = []

	with open(input_file) as file_r:
		reader = csv.DictReader(file_r)
		for row in reader:
			# sets the on_base variables to binary
			on_1 = 0 if row['on_1b'] == "" else 1
			on_2 = 0 if row['on_2b'] == "" else 1
			on_3 = 0 if row['on_3b'] == "" else 1

			new_input = [double(row['hc_x']), double(row['hc_y']), int(row['launch_angle']), double(row['launch_speed']), double(row['estimated_ba_using_speedangle']), int(row['outs_when_up']), row['home_team'], on_1, on_2, on_3, int(row['total_bases'])]
			inputs.append(new_input)
	file_r.close()

	numpy.random.shuffle(inputs)
	split = round(PERCENT_TRAINING * len(inputs))
	training, testing = inputs[:split], inputs[split:]

	train_x = []
	train_y = []
	test_x = []
	test_y = []

	for row in training:
		train_x.append(training[:-1])
		train_y.append(training[-1])
	for row in testing:
		test_x.append(testing[:-1])
		test_y.append(testing[-1])

	train_data = np.asarray(train_x)
	train_labels = np.asarray(train_y)
	eval_data = np.asarray(test_x)
	eval_labels = np.asarray(test_y)

	print("---------------CREATING CLASSIFIER----------------")
	# create estimator
	model = tf.estimator.Estimator(model_fn=neural_net, model_dir = "checkpoints/")

	print("---------------TRAINING CLASSIFIER----------------")
	# train the classifier
	model = train(train_data, train_labels, model)

	print("---------------EVALUATING CLASSIFIER----------------")
	# evaluate effectiveness
	results = test(eval_data, eval_labels, model)
	print(results)

main(input_file)
