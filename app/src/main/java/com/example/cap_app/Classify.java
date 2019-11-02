package com.example.cap_app;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.stream.Stream;

import org.tensorflow.lite.Interpreter;
import android.os.Bundle;
import android.view.Window;
import android.view.WindowManager;

import com.karumi.dexter.Dexter;
import com.karumi.dexter.MultiplePermissionsReport;
import com.karumi.dexter.PermissionToken;
import com.karumi.dexter.listener.PermissionRequest;
import com.karumi.dexter.listener.multi.MultiplePermissionsListener;

import javax.crypto.NoSuchPaddingException;


public class Classify extends AppCompatActivity {

    public static String FILE_NAME_ENC = "";
    public static String FILE_NAME_DEC = "";
    String my_key = "AUIPveKxJ60eb38b";//16char = 128 bits
    String my_spec_key = "g6ly7tFKgCM21v4M";

    // presets for rgb conversion
    private static final int RESULTS_TO_SHOW = 3;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    // options for model interpreter
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    // tflite graph
    private Interpreter tflite;
    // holds all the possible labels for model
    private List<String> labelList;
    // holds the selected image data as bytes
    private ByteBuffer imgData = null;
    // holds the probabilities of each label for non-quantized graphs
    private float[][] labelProbArray = null;
    // holds the probabilities of each label for quantized graphs
    private byte[][] labelProbArrayB = null;
    // array that holds the labels with the highest probabilities
    private String[] topLables = null;
    // array that holds the highest probabilities
    private String[] topConfidence = null;


    // selected classifier information received from extras
    private String chosen;
    //private boolean quant;

    // input image dimensions for the Inception Model
    private int DIM_IMG_SIZE_X = 224;
    private int DIM_IMG_SIZE_Y = 224;
    private int DIM_PIXEL_SIZE = 3;

    // int array to hold image data
    private int[] intValues;

    // activity elements
    private ImageView selected_image;
    private Button classify_button;
    private Button back_button;
    private TextView label1;
    private TextView label2;
    private TextView label3;
    private TextView Confidence1;
    private TextView Confidence2;
    private TextView Confidence3;

    // priority queue that will hold the top results from the CNN
    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    Button btn_enc, btn_dec,btn_select_file;
    ImageView imageView;
    Intent myFileIntent;
    File myDir;
    File FINAL_FILE_NAME_ENC;
    File FINAL_FILE_NAME_DEC;
    Boolean SELECT_USED = false;
    String FINAL_MY_DIR;
    String FINAL_PATH_STRING;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        System.out.println("6. Classify onCreate");
        // get all selected classifier data from classifiers
        chosen = (String) getIntent().getStringExtra("chosen");
        //quant = (boolean) getIntent().getBooleanExtra("quant", false);

        // initialize array that holds image data
        intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

        super.onCreate(savedInstanceState);
        //initilize graph and labels
        try{
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
            labelList = loadLabelList();
        } catch (Exception ex){
            ex.printStackTrace();
        }

        // initialize byte array.
        imgData =
                ByteBuffer.allocateDirect(
                        4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);

        imgData.order(ByteOrder.nativeOrder());

        // initialize probabilities array.
        labelProbArray = new float[1][labelList.size()];

        requestWindowFeature(Window.FEATURE_NO_TITLE); //will hide the title
        getSupportActionBar().hide(); // hide the title bar
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN); //enable full screen
        setContentView(R.layout.activity_classify);

        // labels that hold top three results of CNN
        label1 = (TextView) findViewById(R.id.label1);
        label2 = (TextView) findViewById(R.id.label2);
        label3 = (TextView) findViewById(R.id.label3);
        // displays the probabilities of top labels
        Confidence1 = (TextView) findViewById(R.id.Confidence1);
        Confidence2 = (TextView) findViewById(R.id.Confidence2);
        Confidence3 = (TextView) findViewById(R.id.Confidence3);
        // initialize imageView that displays selected image to the user
        selected_image = (ImageView) findViewById(R.id.selected_image);

        // initialize array to hold top labels
        topLables = new String[RESULTS_TO_SHOW];
        // initialize array to hold top probabilities
        topConfidence = new String[RESULTS_TO_SHOW];

        // allows user to go back to activity to select a different image
        back_button = (Button)findViewById(R.id.back_button);
        back_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(Classify.this, ChooseModel.class);
                startActivity(i);
            }
        });

        // classify current dispalyed image
        classify_button = (Button)findViewById(R.id.classify_image);
        classify_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // get current bitmap from imageView
                Bitmap bitmap_orig = ((BitmapDrawable)selected_image.getDrawable()).getBitmap();
                // resize the bitmap to the required input size to the CNN
                Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
                // convert bitmap to byte array
                convertBitmapToByteBuffer(bitmap);
                // pass byte data to the graph
                tflite.run(imgData, labelProbArray);

                // display the results
                printTopKLabels();
            }
        });

        // get image from previous activity to show in the imageView
        Uri uri = (Uri)getIntent().getParcelableExtra("resID_uri");
        try {
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
            selected_image.setImageBitmap(bitmap);
            // not sure why this happens, but without this the image appears on its side
            selected_image.setRotation(selected_image.getRotation() + 0);
        } catch (IOException e) {
            e.printStackTrace();
        }

        btn_dec = (Button)findViewById(R.id.btn_decrypt);
        btn_enc = (Button)findViewById(R.id.btn_encrypt);
        imageView = (ImageView)findViewById(R.id.selected_image);

        //Init path
        myDir = new File(Environment.getExternalStorageDirectory().toString()+"/Pictures");
        System.out.println("myDir: "+myDir.toString());
        FINAL_MY_DIR = myDir.toString();

        Dexter.withActivity(this)
                .withPermissions(new String[]{
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                })
                .withListener(new MultiplePermissionsListener(){


                    @Override
                    public void onPermissionsChecked(MultiplePermissionsReport report) {
                        btn_dec.setEnabled(true);
                        btn_enc.setEnabled(true);
                    }

                    @Override
                    public void onPermissionRationaleShouldBeShown(List<PermissionRequest> permissions, PermissionToken token) {

                    }
                }).check();

        btn_dec.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if(SELECT_USED==false){
                    FILE_NAME_DEC = getFileName(uri)+".png";
                    System.out.println("FILE_NAME_DEC: "+FILE_NAME_DEC);
                    File delete_junk = new File(myDir,getFileName(uri));
                    File outputFileDec = new File(myDir,FILE_NAME_DEC);
                    System.out.println("outputFileDec: "+outputFileDec.toString());
                    FINAL_FILE_NAME_DEC = outputFileDec;
                    File encFile = new File(myDir, FILE_NAME_ENC);
                    FINAL_FILE_NAME_ENC = encFile;
                    System.out.println("encFile: "+encFile);
                    delete_junk = FINAL_FILE_NAME_ENC;



                    try{
                        System.out.println("FINAL_FILE_NAME_ENC: "+FINAL_FILE_NAME_ENC);
                        System.out.println("FINAL_FILE_NAME_DEC: "+FINAL_FILE_NAME_DEC);
                        MyEncrypter.decryptToFile(my_key,my_spec_key,new FileInputStream(FINAL_FILE_NAME_ENC),
                                new FileOutputStream(FINAL_FILE_NAME_DEC));
                        //After that set for image view
                        selected_image.setImageURI(Uri.fromFile(FINAL_FILE_NAME_DEC));
                        //imageView.setImageURI(Uri.fromFile(outputFileDec));
                        //If you want to delete file after decrypt, keep this
                        delete_junk.delete();
                        Toast.makeText(Classify.this, "Decrypted", Toast.LENGTH_SHORT).show();


                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (NoSuchAlgorithmException e) {
                        e.printStackTrace();
                    } catch (InvalidKeyException e) {
                        e.printStackTrace();
                    } catch (InvalidAlgorithmParameterException e) {
                        e.printStackTrace();
                    } catch (NoSuchPaddingException e) {
                        e.printStackTrace();
                    }
                }else{
                    SELECT_USED = false;
                    FINAL_FILE_NAME_ENC = new File(FINAL_PATH_STRING);
                    FINAL_PATH_STRING = FINAL_PATH_STRING+".png";
                    FINAL_FILE_NAME_DEC = new File(FINAL_PATH_STRING);

                    File delete_junk = FINAL_FILE_NAME_ENC;
                    try{
                        System.out.println("FINAL_FILE_NAME_ENC: "+FINAL_FILE_NAME_ENC);
                        System.out.println("FINAL_FILE_NAME_DEC: "+FINAL_FILE_NAME_DEC);
                        MyEncrypter.decryptToFile(my_key,my_spec_key,new FileInputStream(FINAL_FILE_NAME_ENC),
                                new FileOutputStream(FINAL_FILE_NAME_DEC));
                        //After that set for image view
                        selected_image.setImageURI(Uri.fromFile(FINAL_FILE_NAME_DEC));
                        //imageView.setImageURI(Uri.fromFile(outputFileDec));
                        //If you want to delete file after decrypt, keep this
                        delete_junk.delete();
                        Toast.makeText(Classify.this, "Decrypted", Toast.LENGTH_SHORT).show();


                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (NoSuchAlgorithmException e) {
                        e.printStackTrace();
                    } catch (InvalidKeyException e) {
                        e.printStackTrace();
                    } catch (InvalidAlgorithmParameterException e) {
                        e.printStackTrace();
                    } catch (NoSuchPaddingException e) {
                        e.printStackTrace();
                    }
                }

            }
        });
        btn_enc.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                //Convert drawable to Bitmap
                //BitmapDrawable drawable = (BitmapDrawable) imageView.getDrawable();
                //Bitmap bitmap = drawable.getBitmap();
                System.out.println("Encryption started");
                Bitmap bitmap = null;
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.PNG,100,stream);
                InputStream is = new ByteArrayInputStream(stream.toByteArray());
                System.out.println("Encryption, Image converted");

                //Create file
                FILE_NAME_ENC = getFileName(uri);
                System.out.println("Encryp. file name taken"+FILE_NAME_ENC);
                File outputFileEnc = new File(myDir, FILE_NAME_ENC);
                try{
                    MyEncrypter.encryptToFile(my_key,my_spec_key,is,new FileOutputStream(outputFileEnc));
                    Toast.makeText(Classify.this, "Encrypted", Toast.LENGTH_SHORT).show();
                    selected_image.setImageBitmap(null);
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (NoSuchAlgorithmException e) {
                    e.printStackTrace();
                } catch (InvalidKeyException e) {
                    e.printStackTrace();
                } catch (InvalidAlgorithmParameterException e) {
                    e.printStackTrace();
                } catch (NoSuchPaddingException e) {
                    e.printStackTrace();
                }
            }
        });

        btn_select_file = (Button)findViewById(R.id.select_btn);
        btn_select_file.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                myFileIntent = new Intent(Intent.ACTION_GET_CONTENT);
                myFileIntent.setType("*/*");
                startActivityForResult(myFileIntent,10);
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        switch (requestCode){
            case 10:
                if(resultCode==RESULT_OK){
                    String path = data.getData().getPath();
                    System.out.println(path);
                    String temp[] = path.split("/");
                    path = temp[temp.length-1];
                    path = FINAL_MY_DIR+"/"+path;
                    FINAL_PATH_STRING = path;
                    //FINAL_FILE_NAME_ENC = new File(path);
                    System.out.println("path: "+path);
                    //path = FINAL_MY_DIR+"/"+path+".png";
                    //FINAL_FILE_NAME_DEC = new File(path);
                    SELECT_USED = true;
                }
                break;
        }
    }

    //extract the file name from URI returned from Intent.ACTION_GET_CONTENT?
    public String getFileName(Uri uri) {
        String result = null;
        if (uri.getScheme().equals("content")) {
            Cursor cursor = getContentResolver().query(uri, null, null, null, null);
            try {
                if (cursor != null && cursor.moveToFirst()) {
                    result = cursor.getString(cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME));
                }
            } finally {
                cursor.close();
            }
        }
        if (result == null) {
            result = uri.getPath();
            int cut = result.lastIndexOf('/');
            if (cut != -1) {
                result = result.substring(cut + 1);
            }
        }
        return result;
    }

    // loads tflite grapg from file
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(chosen);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // converts bitmap to byte array which is passed in the tflite graph
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);


            }
        }
    }

    // loads the labels from the label txt file in assets into a string array
    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open("retrained_labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // print the top labels and respective confidences
    private void printTopKLabels() {
        // add all results to priority queue
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));

            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }

        // get top results from priority queue
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            topLables[i] = label.getKey();
            topConfidence[i] = String.format("%.0f%%",label.getValue()*100);
        }

        // set the corresponding textviews with the results
        label1.setText("1. "+topLables[2]);
        if(topLables[2].contains("unsafe")){
            Toast.makeText(Classify.this,"Image is Unsafe, please encrypt it.",Toast.LENGTH_SHORT).show();
        }
        label2.setText("2. "+topLables[1]);
        label3.setText("3. "+topLables[0]);
        Confidence1.setText(topConfidence[2]);
        Confidence2.setText(topConfidence[1]);
        Confidence3.setText(topConfidence[0]);
    }


    // resizes bitmap to given dimensions
    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }
}
