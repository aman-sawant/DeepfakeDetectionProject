package com.dfd;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import okhttp3.*;

import java.io.File;
import java.io.IOException;

public class Main extends Application {
    private final OkHttpClient client = new OkHttpClient();

    private Label resultLabel = new Label();
    private ProgressIndicator spinner = new ProgressIndicator();
    private ImageView imageView = new ImageView();

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) {
        // Buttons
        Button audioBtn = new Button("🎵 Upload Audio");
        Button imageBtn = new Button("🖼️ Upload Image");

        audioBtn.setStyle("-fx-font-size: 16px; -fx-background-color: #4CAF50; -fx-text-fill: white;");
        imageBtn.setStyle("-fx-font-size: 16px; -fx-background-color: #2196F3; -fx-text-fill: white;");

        audioBtn.setOnAction(e -> chooseAndSend(
                "WAV Files",
                new FileChooser.ExtensionFilter[] {
                        new FileChooser.ExtensionFilter("WAV", "*.wav")
                },
                "audio/wav",
                "http://127.0.0.1:5000/predict-audio"
        ));

        imageBtn.setOnAction(e -> chooseAndSend(
                "Image Files",
                new FileChooser.ExtensionFilter[] {
                        new FileChooser.ExtensionFilter("JPEG", "*.jpg"),
                        new FileChooser.ExtensionFilter("PNG", "*.png")
                },
                "image/jpeg",
                "http://127.0.0.1:5000/predict-image"
        ));

        // Tooltips for buttons
        audioBtn.setTooltip(new Tooltip("Click to upload an audio file for analysis"));
        imageBtn.setTooltip(new Tooltip("Click to upload an image for analysis"));

        // UI Layout
        spinner.setVisible(false);
        spinner.setStyle("-fx-progress-color: #FF5722;");
        imageView.setFitWidth(300);
        imageView.setPreserveRatio(true);
        imageView.setStyle("-fx-effect: dropshadow(gaussian, rgba(0, 0, 0, 0.7), 10, 0.5, 0, 0);");

        resultLabel.setStyle("-fx-text-fill: white; -fx-font-size: 18; -fx-font-weight: bold;");

        VBox root = new VBox(20,
                audioBtn, imageBtn,
                spinner, imageView,
                resultLabel
        );
        root.setAlignment(Pos.CENTER);
        root.setStyle("-fx-background-color: #2E2E2E; -fx-padding: 30;");
        root.setPrefWidth(600);
        root.setPrefHeight(550);
        root.setSpacing(20);

        HBox buttonBox = new HBox(20, audioBtn, imageBtn);
        buttonBox.setAlignment(Pos.CENTER);
        buttonBox.setStyle("-fx-padding: 10;");

        VBox mainLayout = new VBox(20, buttonBox, spinner, imageView, resultLabel);
        mainLayout.setAlignment(Pos.CENTER);
        mainLayout.setStyle("-fx-background-color: #2E2E2E; -fx-padding: 30;");

        stage.setScene(new Scene(mainLayout));
        stage.setTitle("Deepfake Audio & Image Detector");
        stage.show();
    }

    private final void chooseAndSend(String title, FileChooser.ExtensionFilter[] filters, String mediaType, String url) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle(title);
        chooser.getExtensionFilters().addAll(filters);
        File file = chooser.showOpenDialog(null);
        if (file == null) return;

        resultLabel.setText("Processing...");
        spinner.setVisible(true);

        // If image, show preview immediately
        if (mediaType.startsWith("image")) {
            imageView.setImage(new Image(file.toURI().toString(), 300, 0, true, true));
        } else {
            imageView.setImage(null);
        }

        new Thread(() -> sendFile(file, mediaType, url)).start();
    }

    private void sendFile(File file, String mediaType, String url) {
        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", file.getName(),
                        RequestBody.create(file, MediaType.parse(mediaType)))
                .build();

        Request req = new Request.Builder()
                .url(url)
                .post(body)
                .build();

        try (Response res = client.newCall(req).execute()) {
            String text = res.body().string();
            if (!res.isSuccessful()) {
                updateUI("Server error: " + res.code() + "\n" + text);
                return;
            }
            JsonObject json = JsonParser.parseString(text).getAsJsonObject();
            String pred = json.get("prediction").getAsString();
            String conf = json.get("confidence").getAsString();
            updateUI("🔍 Prediction: " + pred + "\n✅ Confidence: " + conf);
        } catch (IOException e) {
            updateUI("Error: " + e.getMessage());
        }
    }

    private void updateUI(String msg) {
        Platform.runLater(() -> {
            resultLabel.setText(msg);
            spinner.setVisible(false);
        });
    }
}
