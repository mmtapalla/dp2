# Add Coral Edge TPU repository to the package sources list
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Download and add GPG key for the Coral Edge TPU repository
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update package lists install the TFLite runtime
sudo apt update
sudo apt install python3-tflite-runtime

# Clone TF examples repository
git clone https://github.com/tensorflow/examples.git --depth 1