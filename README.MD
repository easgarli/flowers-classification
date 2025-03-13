# Flower Classification Flask App

This Flask application classifies images of flowers into five categories: daisy, dandelion, roses, sunflowers, and tulips. The app allows users to upload a flower image, choose a model (Custom CNN, VGG16, or ResNet152), and view the classification results along with the confidence score.

## Features

- **Model Selection**: Choose between Custom CNN, VGG16, and ResNet152 models
- **Image Upload**: Upload a flower image (JPEG or PNG formats)
- **Classification**: Classifies the uploaded image into one of the flower categories
- **Confidence Score**: Displays the model's confidence in the classification
- **RESTful API**: Provides a simple API endpoint for predictions

## Project Structure

```
├── app.py                         # Main Flask application
├── custom_cnn_model_savedmodel    # Custom CNN model files
│   ├── fingerprint.pb
│   └── saved_model.pb
├── templates                      # HTML templates
│   └── index.html
├── gunicorn_config.py            # Gunicorn configuration
├── requirements.txt              # Python dependencies
├── ResNet152_savedmodel         # ResNet152 model files
│   ├── fingerprint.pb
│   └── saved_model.pb
└── VGG16_savedmodel            # VGG16 model files
    ├── fingerprint.pb
    └── saved_model.pb
```

## Local Development Setup

1. **Clone the repository**:
```
```
   git clone https://github.com/easgarli/flower-classification-app.git
   cd flower-classification-app
```
```

## Create a virtual environment (recommended):

```
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

## Install dependencies:

```
pip install -r requirements.txt
```

## Run the Flask app locally:

```
python app.py
```

## Production Deployment

### Prerequisites
- Ubuntu 24.04 LTS server
- Domain name pointed to your server
- Python 3.8+
- Nginx
- Gunicorn

### Deployment Steps

1. **Install system dependencies**:
```
sudo apt update
sudo apt install python3-pip python3-venv nginx
```

2. **Set up application directory**:
```
sudo mkdir -p /var/www/flower-classification
sudo chown $USER:$USER /var/www/flower-classification
```

3. **Copy application files**:
```
cp -r * /var/www/flower-classification/
```

4. **Create virtual environment and install requirements**:
```
cd /var/www/flower-classification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. **Create systemd service**:
```
sudo nano /etc/systemd/system/flower-classification.service
```

Add the following content:
```ini
[Unit]
Description=Gunicorn instance to serve flower classification app
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/flower-classification
Environment="PATH=/var/www/flower-classification/venv/bin"
ExecStart=/var/www/flower-classification/venv/bin/gunicorn -c gunicorn_config.py app:app

[Install]
WantedBy=multi-target.target
```

6. **Configure Nginx**:
```
sudo nano /etc/nginx/sites-available/flower-classification
```

Add the following content:
```nginx
server {
    listen 80;
    server_name image-classification.elnurdevhub.com;

    location / {
        proxy_pass http://unix:/tmp/flower_classification.sock;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

7. **Set up permissions and enable services**:
```
sudo chown -R www-data:www-data /var/www/flower-classification
sudo ln -s /etc/nginx/sites-available/flower-classification /etc/nginx/sites-enabled/
sudo systemctl enable flower-classification
sudo systemctl start flower-classification
sudo systemctl restart nginx
```

8. **Set up SSL (recommended)**:
```
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d image-classification.elnurdevhub.com
```


## API Usage

### Endpoint: `/predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `file`: Image file (jpg, jpeg, or png)
  - `model`: Model choice (custom_cnn, vgg16, or resnet152)
- **Response**: JSON
{
"prediction": "class_name",
"confidence": 0.95
}

## Dependencies
- Flask 3.0.2
- Gunicorn 21.2.0
- NumPy 1.24.3
- OpenCV (Headless) 4.9.0.80
- Pillow 10.2.0
- TensorFlow 2.15.0

## Troubleshooting

1. **Model Loading Issues**:
   - Ensure all model files are in the correct directories
   - Check file permissions

2. **Server Issues**:
   - Check logs: `sudo journalctl -u flower-classification`
   - Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`

3. **Permission Issues**:
   - Ensure proper ownership: `sudo chown -R www-data:www-data /var/www/flower-classification`
   - Check socket permissions: `ls -l /tmp/flower_classification.sock`

## License
This project is licensed under the MIT License.
