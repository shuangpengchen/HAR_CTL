package com.example.har_ctl;
import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static java.lang.System.exit;

public class stage2 extends AppCompatActivity implements SensorEventListener {

    private Button btn1,btn2;
    private TextView pridiction,status,sensor;

    private final String  modelPath = "model_v91.tflite";
    private final String  labelPath = "label_v1.txt";
    private String[] labels = {"Forward", "Backward", "Turn Left", "Turn Right", "Stop","Still"};
//    private final String dataPath = "a1.csv";

    private final int N_TIME_STEP = 150;
    private final int N_FEATURE = 3;


    private TextToSpeech textToSpeech;
    private CTLClassifier classifier;

    private float[][] input;
    private float[] results;
    private int counter ;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_stage2);
        btn1 = findViewById(R.id.s2start);
        btn2 = findViewById(R.id.s2end);
        pridiction = findViewById(R.id.s2predictiondata);
        status = findViewById(R.id.s2status);
        sensor = findViewById(R.id.s2sensordata);
        input = new float[this.N_TIME_STEP][this.N_FEATURE];
        counter = 0;


        try {
            classifier = CTLClassifier.create(getAssets(), modelPath, labelPath);
        } catch (IOException e) {
            throw new RuntimeException("Error initializing TensorFlow!", e);
        }
        this.textToSpeech = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR) {
                    textToSpeech.setLanguage(Locale.US);
                }else{
                    System.out.println(status);
                    exit(-1);
                }
            }
        });

    }

//    private void split_data_into_inputs(){
//        int count=0;
//        int pointer=0;
//        float[][] temp = new float[this.N_TIME_STEP][this.N_FEATURE];
//        for(int i=0;i<n_rounds*this.N_TIME_STEP;i++){
//            temp[pointer++] = this.raw_data.get(i);
//
//            if((pointer)%this.N_TIME_STEP == 0){
//                count++;
//                this.inputs.add(temp);
//                pointer=0;
//                temp = new float[this.N_TIME_STEP][this.N_FEATURE];
//            }
//        }
//        System.out.println("raw data size is: "+this.raw_data.size());
//        System.out.println("inputs size is : "+ this.inputs.size());
//    }

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }


    public void start(View v){
        status.setText("Status: Running");
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME);
    }

//    private void predict_evaldata(int known_act){
//        float count=0;
//        for(int i=0;i<this.inputs.size();i++){
//            System.out.println(i);
//            float[] temp_result = this.classifier.predict((this.inputs.get(i)));
//            if( (maxIdx(temp_result)+1) == known_act){
//                count++;
//            }
//        }
//        System.out.println("count : " + count);
//        System.out.println("inputs size: "+this.inputs.size());
//        double acc = count / (float)this.inputs.size();
//        System.out.println("FINAL ACC : " + acc);
//    }


    private void shift(){
        for(int i=0;i<counter;i++){
            this.input[i] = this.input[i+counter];
        }
    }

    private void predict(float x,float y,float z){
        if(this.counter == this.N_TIME_STEP ){
            System.out.println("predicting");
            results = this.classifier.predict(this.input);
            String message = this.labels[this.maxIdx(results)];
            this.textToSpeech.speak(message,TextToSpeech.QUEUE_FLUSH, null);
            input = new float[this.N_TIME_STEP][this.N_FEATURE];
            counter = counter/2;
            this.shift();
        }
        this.input[counter] = new float[]{x, y, z};
        counter++;
    }

    public void end(View v){
        status.setText(("Status: Stopped"));
        getSensorManager().unregisterListener(this);
        this.classifier.close();
        finish();
    }


    private int maxIdx(float[] result){
        if( result == null || result.length == 0){
            return -1;
        }
        int idx=0;
        float max=-1000;
        for(int i=0;i< result.length;i++){
            if(result[i] > max){
                max = result[i];
                idx = i;
            }
        }
        return idx;
    }


    private ArrayList<float[]> loadDataset(AssetManager assetManager, String evalDataPath) throws IOException {
        ArrayList<float[]> dataset = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(evalDataPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            float[] temp = new float[3];
            temp[0] = Float.parseFloat(parts[2]);
            temp[1] = Float.parseFloat(parts[3]);
            temp[2] = Float.parseFloat(parts[4]);
            dataset.add(temp);
        }
        reader.close();
        System.out.print("dataset is ready......");
        return dataset;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        predict(event.values[0],event.values[1],event.values[2]);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}



