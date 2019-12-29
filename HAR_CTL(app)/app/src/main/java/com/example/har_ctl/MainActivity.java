package com.example.har_ctl;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {


    Button btn1;
    Button btn2;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn1 = findViewById(R.id.button1);
        btn2 = findViewById(R.id.button2);
    }


    public void go2Stage1(View v){
        Intent intent = new Intent(this,stage1.class);
        startActivity(intent);
    }

    public void go2Stage2(View v){
        Intent intent = new Intent(this,stage2.class);
        startActivity(intent);
    }



}
