import tensorflow as tf
import numpy
import csv

input_files = ['processed_data_2015.csv', 'processed_data_2016.csv', 'processed_data_2017.csv', 'processed_data_2018.csv']
PERCENT_TRAINING = 0.8
BATCH_SIZE = 200
STEPS = 10000
LEARNING_RATE = 0.001
NUM_EPOCHS = 100



def neural_net(features, labels, mode):

	print("Starting NN")

	dense_layer_1 = tf.layers.dense(inputs=features, units=20, activation=tf.nn.relu)
	dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=20, activation=tf.nn.relu)
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
		x=train_data,
		y=train_labels,
		batch_size=BATCH_SIZE,
		num_epochs=NUM_EPOCHS,
		shuffle=True)
	classifier.train(
		input_fn=train_input_fn,
		steps=STEPS,
		hooks=[logging_hook])

	return classifier

def test(eval_data, eval_labels, classifier):
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    	x=eval_data,
    	y=eval_labels,
    	num_epochs=1,
    	shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	eval_predictions = classifier.predict(input_fn=eval_input_fn)
	
	return eval_results, eval_predictions


def main(input_files):
	inputs = []

	team_dict = {"ARI": 0, "ATL": 1, "BAL": 2, "BOS": 3, "CHC": 4, "CWS": 5, "CIN": 6, "CLE": 7, "COL": 8, "DET": 9, "MIA": 10, "HOU": 11, "KC": 12, "LAA": 13, "LAD":14, "MIL":15, "MIN":16, "NYM": 17, "NYY":18, "OAK":19, "PHI":20, "PIT": 21, "SD": 22, "SF": 23, "SEA": 24, "STL": 25, "TB": 26, "TEX": 27, "TOR": 28, "WSH": 29 }

	print("begin")

	for input_file in input_files:
		with open(input_file) as file_r:
			reader = csv.DictReader(file_r)
			missing = 0
			for row in reader:
				# sets the on_base variables to binary
				on_1 = 0 if row['on_1b'] == "" else 1
				on_2 = 0 if row['on_2b'] == "" else 1
				on_3 = 0 if row['on_3b'] == "" else 1

				if row['hc_x'] == "" or row['hc_y'] == "" or row['launch_angle'] == "" or row['launch_speed'] == "" or row['estimated_ba_using_speedangle'] == "" or row['outs_when_up'] == "" or row['total_bases'] == "": # add or row['home_team'] == ""
					missing += 1
				else:
					new_input = [float(row['hc_x']), float(row['hc_y']), float(row['launch_angle']), float(row['launch_speed']), float(row['estimated_ba_using_speedangle']), int(float(row['outs_when_up'])), on_1, on_2, on_3,  int(float(row['total_bases']))] # row['home_team'],  # team_dict[row['home_team']],  
					inputs.append(new_input)
					if int(float(row['total_bases'])) > 6:
						print("Above 6", int(float(row['total_bases'])))
		file_r.close()

		print(input_file, "missing data:", missing)

		print("closed file")

	numpy.random.shuffle(inputs)

	print("completed shuffle")
	split = round(PERCENT_TRAINING * len(inputs))
	training, testing = inputs[:split], inputs[split:]

	print("completed split")

	train_x = []
	train_y = []
	test_x = []
	test_y = []

	for row in training:
		train_x.append(row[:-1])
		train_y.append(row[-1] + 6)	# shifted by 6 bases, CHANGE THIS!!!
	for row in testing:
		test_x.append(row[:-1])
		test_y.append(row[-1] + 6) # shifted by 6 bases, CHANGE THIS!!!

	print("completed x/y var separation")

	train_data = numpy.asarray(train_x)
	train_labels = numpy.asarray(train_y)
	eval_data = numpy.asarray(test_x)
	eval_labels = numpy.asarray(test_y)

	print("---------------CREATING CLASSIFIER----------------")
	# create estimator
	model = tf.estimator.Estimator(model_fn=neural_net, model_dir = "checkpoints/")

	print("---------------TRAINING CLASSIFIER----------------")
	# train the classifier
	model = train(train_data, train_labels, model)

	print("---------------EVALUATING CLASSIFIER----------------")
	# evaluate effectiveness
	results, pred_gen = test(eval_data, eval_labels, model)
	print(results)
	#print(test_y)
	# print(pred)
	pred_gen = list(pred_gen)
	predict = []

	for row in pred_gen:
		predict.append(row["classes"])

	#print(pred)

	# matrix = tf.math.confusion_matrix(test_y, predict)
	# print(matrix)
	with tf.Session() as sess:
		confusion_matrix = tf.confusion_matrix(labels=test_y, predictions=predict)
		confusion_matrix_to_Print = sess.run(confusion_matrix)
		print(confusion_matrix_to_Print)


main(input_files)
