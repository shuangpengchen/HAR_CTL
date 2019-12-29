Project: HAR_CTL
Description: 
			1.HAR-human activity recognition; 
			2.CTL-control activity recognition;
Data:	
	HAR data is download from http://www.cis.fordham.edu/wisdm/dataset.php
	CTL data is collected by myself

Test Device:
	Nexus 6P

Models:
	v1:
		Dense -> LSTM -> LSTM -> Dense
		status: failed due to tflite doesn't support partial ops
	v2:
		Conv1D -> Dropout -> Maxpooling1D -> Flatten -> Dense
		status: successed

Result:
	CTL:
		test acc:	0.9998304843902588	
		test lose:	0.0010805797882609334
	HAR:
		test acc:	0.9923349022865295
		test lose:	0.02775145718281109

Files Structures:
	1.trainning:
		folders:
				1.CTL_checkpoints:	checkpoints dir for CTL
				2.CTL_models:		contain CTL models in h5 format and model info
				3.CTL_tflites:		CTL models in tflite format
				4.data:				trainning data
				5.HAR_checkpoints:	checkpoints dir for HAR
				6.HAR_models:		contain HAR models in h5 format and model info
				7.HAR_tflites:		HAR models in tflite format
		files:
				1.CTL_(..time..)_log.txt:	trainning log for each models for CTL
				2.HAR_(..time..)_log.txt:	trainning log for each models for HAR
				3.train.py:					main code for trainning
	2. HAR_CTL(app):
		android app
	3. Demo:
		demo video

Conclusion:
	CTL:
		Status: Work as expected
	HAR:
		Status: Did not work as expected
		Possiable Reasones: 
			1. unknown tranning position(where to place the device).
			2. unreliable dataset
