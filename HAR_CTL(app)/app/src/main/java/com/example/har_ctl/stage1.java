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
import java.util.Locale;

import static java.lang.System.exit;

public class stage1 extends AppCompatActivity implements SensorEventListener {


    private TextToSpeech textToSpeech;
    private HARClassifier classifier;

    private float[][] input;
    private float[] results;
    private int counter;

    private final int N_TIME_STEP = 170;
    private final int N_FEATURE = 3;

    private Button start,end;
    private TextView prediction,status;

    private final String  modelPath = "model_v98.tflite";
    private final String  labelPath = "label_v1.txt";

    private final String[] labels = { "Downstairs","Jogging", "Sitting", "Standing","Upstairs","Walking" };

    private final int WINDOW_SIZE = 170;
    private int version = 1;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_stage1);
        start = findViewById(R.id.s1start);
        end = findViewById(R.id.s1end);
        prediction = findViewById(R.id.predictiondata);
        status = findViewById(R.id.s1status);
        input = new float[this.N_TIME_STEP][this.N_FEATURE];
        counter = 0;


        try {
            classifier = HARClassifier.create(getAssets(), modelPath, labelPath);
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

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }


    public void start(View v){
        status.setText("Status: Running");
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME);
    }


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


