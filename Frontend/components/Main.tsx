import React, { useState } from "react";
// import { CVImage, InferenceEngine } from "inferencejs";
import {
  View,
  Text,
  Button,
  ActivityIndicator,
  Image,
  StyleSheet,
} from "react-native";
import * as ImagePicker from "expo-image-picker"; // Import the image picker

export default function Main() {
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string>(""); // State for the selected image

  const handleCameraUpload = async () => {
    // Request permission to access the camera
    const cameraPermission = await ImagePicker.requestCameraPermissionsAsync();
    if (!cameraPermission.granted) {
      alert("Permission to access the camera is required!");
      return;
    }
    // Launch the camera
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri); // Set the captured image URI
      console.log("Captured image URI:", result.assets[0].uri);
      uploadImage(result.assets[0].uri);
    } else {
      console.log("Camera action canceled.");
    }
  };
  const handleImageUpload = async () => {
    // Request permission to access the media library
    const galleryPermission =
      await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!galleryPermission.granted) {
      alert("Permission to access the gallery is required!");
      return;
    }

    // Launch the image picker
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri); // Set the selected image URI
      console.log("Selected image URI:", result.assets[0].uri);
      uploadImage(result.assets[0].uri);
    } else {
      console.log("Image selection canceled.");
    }
  };

  const uploadImage = async (imageFile: string) => {
    const formData = new FormData();
    formData.append("image", {
      uri: imageFile,
      name: "Test.PNG",
      type: "image/png",
    } as any);

    try {
      const response = await fetch("http://192.168.18.2:5010/detect", {
        method: "POST",
        headers: {
          "Content-Type": "multipart/form-data",
        },
        body: formData,
      });

      // const result = await response.text(); // or .text() if it's not JSON
      const result = await response.text();
      console.log("Detection result:", result);
      setResponse(result);
    } catch (error) {
      console.error("Upload error:", error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Garbage Classifier</Text>

      {selectedImage ? (
        <Image
          source={{ uri: selectedImage }}
          style={{ width: 200, height: 200 }}
        />
      ) : (
        <Text>No image selected</Text>
      )}

      <>
        <Button title="Camera" onPress={handleCameraUpload} />
        <Button title="Upload Image" onPress={handleImageUpload} />
      </>

      {selectedImage && loading ? (
        <ActivityIndicator
          size="large"
          color="#0000ff"
          style={styles.loading}
        />
      ) : response ? (
        <Text style={styles.description}>{response}</Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 24,
    paddingTop: 60,
    flex: 1,
    backgroundColor: "#fff",
  },
  heading: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 12,
  },
  description: {
    fontSize: 12,
    marginBottom: 12,
  },
  input: {
    borderColor: "#ccc",
    borderWidth: 1,
    padding: 12,
    fontSize: 16,
    marginBottom: 12,
    borderRadius: 8,
    minHeight: 80,
  },
  loading: {
    marginTop: 20,
  },
  responseContainer: {
    marginTop: 20,
  },
  response: {
    fontSize: 16,
    lineHeight: 24,
  },
});
