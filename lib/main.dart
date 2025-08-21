import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const CattleWeightApp());
}

class CattleWeightApp extends StatelessWidget {
  const CattleWeightApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Cattle Weight Estimator',
      theme: ThemeData(
        primarySwatch: Colors.green,
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.green,
          brightness: Brightness.light,
        ),
      ),
      home: const CattleWeightHomePage(),
    );
  }
}

class CattleWeightHomePage extends StatefulWidget {
  const CattleWeightHomePage({super.key});

  @override
  CattleWeightHomePageState createState() => CattleWeightHomePageState();
}

class CattleWeightHomePageState extends State<CattleWeightHomePage> {
  File? _image;
  double? _predictedWeight;
  bool _isLoading = false;
  Interpreter? _interpreter;
  String? _errorMessage;
  bool _showHint = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }
Future<void> _loadModel() async {
  try {
    debugPrint("Loading model from: assets/cow_weight_model.tflite");
    _interpreter = await Interpreter.fromAsset("assets/cow_weight_model.tflite");
    debugPrint("Model loaded! Input shape: ${_interpreter!.getInputTensor(0).shape}");
    debugPrint("Output shape: ${_interpreter!.getOutputTensor(0).shape}");
    setState(() {});
  } catch (e, stackTrace) {
    _showError("Failed to load model: $e");
    debugPrint("Model load error: $e\n$stackTrace");
  }
}


  Future<void> _pickImage(ImageSource source) async {
    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(source: source, imageQuality: 80); // Reduce image quality for performance
      if (pickedFile == null) return;

      setState(() {
        _image = File(pickedFile.path);
        _predictedWeight = null;
        _isLoading = true;
        _errorMessage = null;
      });

      await _predictWeight();
    } catch (e) {
      _showError('Error picking image: $e');
      setState(() => _isLoading = false);
    }
  }

 Future<void> _predictWeight() async {
  if (_interpreter == null || _image == null) {
    _showError('Model or image not loaded');
    setState(() => _isLoading = false);
    return;
  }

    try {
      // Load and preprocess image
      final imageBytes = await _image!.readAsBytes();
      img.Image? image = img.decodeImage(imageBytes);
      if (image == null) {
        _showError('Failed to decode image');
        setState(() => _isLoading = false);
        return;
      }

      // Resize to 128x128
      image = img.copyResize(image, width: 128, height: 128, interpolation: img.Interpolation.average);

      // Prepare input tensor (1, 128, 128, 3)
      final input = Float32List(1 * 128 * 128 * 3);
      var index = 0;
      for (var y = 0; y < 128; y++) {
        for (var x = 0; x < 128; x++) {
          final pixel = image.getPixel(x, y);
          input[index++] = pixel.r / 255.0; // Red
          input[index++] = pixel.g / 255.0; // Green
          input[index++] = pixel.b / 255.0; // Blue
        }
      }
      final inputReshaped = input.reshape([1, 128, 128, 3]);

      // Prepare output tensor (1, 1)
      final output = List.filled(1, 0.0).reshape([1, 1]);

      // Run inference
      _interpreter!.run(inputReshaped, output);

      // Update UI with prediction
      setState(() {
        _predictedWeight = output[0][0];
        _isLoading = false;
      });
    } catch (e) {
      _showError('Error during prediction: $e');
      setState(() => _isLoading = false);
    }
  }

  void _showError(String message) {
    setState(() => _errorMessage = message);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 3),
      ),
    );
  }

  void _toggleHint() {
    setState(() {
      _showHint = !_showHint;
    });
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cattle Weight Estimator'),
        centerTitle: true,
        backgroundColor: Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).colorScheme.primary.withOpacity(0.1),
              Colors.white,
            ],
          ),
        ),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      _image == null
                          ? Container(
                              height: 250,
                              width: double.infinity,
                              decoration: BoxDecoration(
                                color: Colors.grey[200],
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: const Center(
                                child: Icon(
                                  Icons.camera_alt,
                                  size: 50,
                                  color: Colors.grey,
                                ),
                              ),
                            )
                          : ClipRRect(
                              borderRadius: BorderRadius.circular(12),
                              child: Image.file(
                                _image!,
                                height: 250,
                                width: double.infinity,
                                fit: BoxFit.cover,
                              ),
                            ),
                      const SizedBox(height: 20),
                      if (_isLoading)
                        const CircularProgressIndicator()
                      else if (_errorMessage != null)
                        Text(
                          _errorMessage!,
                          style: const TextStyle(color: Colors.red, fontSize: 16),
                          textAlign: TextAlign.center,
                        )
                      else if (_predictedWeight != null)
                        Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: Colors.green.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            'Estimated Weight: ${_predictedWeight!.toStringAsFixed(2)} kg',
                            style: const TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: Colors.green,
                            ),
                          ),
                        )
                      else
                        const Text(
                          'Select or capture an image to predict weight',
                          style: TextStyle(fontSize: 16),
                          textAlign: TextAlign.center,
                        ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Take Photo'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                      backgroundColor: Theme.of(context).colorScheme.primary,
                      foregroundColor: Colors.white,
                    ),
                  ),
                  const SizedBox(width: 20),
                  ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Pick from Gallery'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                      backgroundColor: Theme.of(context).colorScheme.secondary,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              ElevatedButton.icon(
                onPressed: _toggleHint,
                icon: Icon(_showHint ? Icons.visibility_off : Icons.visibility),
                label: Text(_showHint ? 'Hide Hint' : 'Show Hint'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                  backgroundColor: Colors.orange,
                  foregroundColor: Colors.white,
                ),
              ),
              if (_showHint)
                Container(
                  margin: const EdgeInsets.only(top: 20),
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.orange.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.orange),
                  ),
                  child: Column(
                    children: const [
                      Text(
                        'Photo Taking Instructions:',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.orange,
                        ),
                      ),
                      SizedBox(height: 10),
                      Text(
                        '1. Stand approximately 1.5-2 meters away from the cattle\n'
                        '2. Place a 10cm green square reference marker on the cattle\'s body\n'
                        '3. Ensure good lighting and clear visibility\n'
                        '4. Capture the entire body of the cattle in the frame',
                        style: TextStyle(fontSize: 16),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}