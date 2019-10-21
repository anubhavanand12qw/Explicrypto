package com.example.cap_app;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.os.Bundle;
import android.view.Window;
import android.view.WindowManager;
//import android.support.v7.app.AppCompatActivity;

import com.soundcloud.android.crop.Crop;

import java.io.File;

import androidx.appcompat.app.AppCompatActivity;

public class ChooseModel extends AppCompatActivity {
    // button for each available classifier
    private Button inceptionFloat;
    private Button inceptionQuant;

    // for permission requests
    public static final int REQUEST_PERMISSION = 300;

    // request code for permission requests to the os for image
    public static final int REQUEST_IMAGE = 100;

    // will hold uri of image obtained from camera
    private Uri imageUri;

    // string to send to next activity that describes the chosen classifier
    private String chosen;

    //boolean value dictating if chosen model is quantized version or not.
    //private boolean quant;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        System.out.println("1. Starting first !!");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE); //will hide the title
        getSupportActionBar().hide(); // hide the title bar
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN); //enable full screen
        setContentView(R.layout.activity_choose_model);

        // request permission to use the camera on the user's phone
        if (ActivityCompat.checkSelfPermission(this.getApplicationContext(), android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, new String[] {android.Manifest.permission.CAMERA}, REQUEST_PERMISSION);
        }

        // request permission to write data (aka images) to the user's external storage of their phone
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_PERMISSION);
        }

        // request permission to read data (aka images) from the user's external storage of their phone
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    REQUEST_PERMISSION);
        }

        // on click for inception float model
        inceptionFloat = (Button)findViewById(R.id.inception_float);
        inceptionFloat.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // filename in assets
                chosen = "optimized_graph.lite";
                // model in not quantized
                //quant = false;
                // open camera
                System.out.println("2. Opening camera intent");
                openCameraIntent();
            }
        });

        // on click for inception quant model
        /*
        inceptionQuant = (Button)findViewById(R.id.inception_quant);
        inceptionQuant.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // filename in assets
                chosen = "optimized_graph.lite";
                // model in not quantized
                quant = true;
                // open camera
                openCameraIntent();
            }
        });*/
    }

    // opens camera for user
    private void openCameraIntent(){
        System.out.println("3. Open camera intent");
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, "New Picture");
        values.put(MediaStore.Images.Media.DESCRIPTION, "From your Camera");
        // tell camera where to store the resulting picture
        imageUri = getContentResolver().insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        // start camera, and wait for it to finish
        startActivityForResult(intent, REQUEST_IMAGE);
    }

    // checks that the user has allowed all the required permission of read and write and camera. If not, notify the user and close the application
    @Override
    public void onRequestPermissionsResult(final int requestCode, @NonNull final String[] permissions, @NonNull final int[] grantResults) {
        System.out.println("4. onRequestPermissionsResult");
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION) {
            if (!(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                Toast.makeText(getApplicationContext(),"This application needs read, write, and camera permissions to run. Application now closing.", Toast.LENGTH_LONG);
                System.exit(0);
            }
        }
    }

    // dictates what to do after the user takes an image, selects and image, or crops an image
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        System.out.println("5. onActivityResult");
        System.out.println("5.1. onActivityResult");
        Intent i = new Intent(ChooseModel.this, Classify.class);
        // put image data in extras to send
        i.putExtra("resID_uri", imageUri);
        // put filename in extras
        i.putExtra("chosen", chosen);
        // put model type in extras
        //i.putExtra("quant", quant);
        // send other required data
        startActivity(i);
        setContentView(R.layout.activity_classify);

        // if the camera activity is finished, obtained the uri, crop it to make it square, and send it to 'Classify' activity
        /*if(requestCode == REQUEST_IMAGE && resultCode == RESULT_OK) {
            try {
                Uri source_uri = imageUri;
                Uri dest_uri = Uri.fromFile(new File(getCacheDir(), "cropped"));
                // need to crop it to square image as CNN's always required square input
                Crop.of(source_uri, dest_uri).asSquare().start(ChooseModel.this);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }*/

        // if cropping acitivty is finished, get the resulting cropped image uri and send it to 'Classify' activity
        /*else if(requestCode == Crop.REQUEST_CROP && resultCode == RESULT_OK){
            System.out.println("5.1. onActivityResult");
            imageUri = Crop.getOutput(data);
            Intent i = new Intent(ChooseModel.this, Classify.class);
            // put image data in extras to send
            i.putExtra("resID_uri", imageUri);
            // put filename in extras
            i.putExtra("chosen", chosen);
            // put model type in extras
            //i.putExtra("quant", quant);
            // send other required data
            startActivity(i);
            setContentView(R.layout.activity_classify);
        }*/
    }
}
