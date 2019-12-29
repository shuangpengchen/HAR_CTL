package com.example.har_ctl;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CTLClassifier {

    private Interpreter interpreter;
    private List<String> labelList;
    private final int classes=6;

    public CTLClassifier() {}


    static CTLClassifier create(AssetManager assetManager, String modelPath, String labelPath) throws IOException {
        CTLClassifier classifier = new CTLClassifier();
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath), new Interpreter.Options());
        //classifier.labelList = classifier.loadLabelList(assetManager, labelPath);
        return classifier;
    }


    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    public float[] predict(float[][] input) {
        float[][] result = new float[1][classes];
        interpreter.run(input, result);
        System.out.println("Input : "+Arrays.toString(input[0]));
        System.out.println("Output : "+Arrays.toString(result[0]));
        return result[0];
    }

    public void close(){
        this.interpreter.close();
    }


}
