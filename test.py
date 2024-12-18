import requests
import base64
from PIL import Image
from io import BytesIO
 

def image_to_base64(orig_image):
    image = BytesIO(orig_image)
    img_str = base64.b64encode(image.getvalue()).decode("utf-8")
    return img_str


def local_image_to_bytes(path:str, img_format="PNG", compress_level:int=9) -> bytes:
    img = Image.open(path)
    with BytesIO() as buff:
        # save png file to buff
        img.save(buff, format=img_format, compress_level=compress_level)
        # get bytes
        buff.seek(0) 
        out = buff.read()
    return out


def image_from_url_to_base64(image_url):
    response = requests.get(image_url)
    orig_image = response.content
    base64_img = image_to_base64(orig_image)
    return base64_img


def image_from_local_path_to_base64(image_path, image_format):
    img_bytes = local_image_to_bytes(image_path, image_format)
    base64_img = image_to_base64(img_bytes)
    return base64_img


def post_image_to_api(image):
    #url = 'http://localhost:9000/2015-03-31/functions/function/invocations'
    url = 'https://8t510ynh79.execute-api.us-east-1.amazonaws.com/test/predict'
    data = {'image': image}
    result = requests.post(url, json=data).json()
    return result


image_url = 'https://www.ultrasoundcases.info/clients/ultrasoundcases/uploads/61570-Afbeelding1.jpg'
image1 = image_from_url_to_base64(image_url)
#print(image1)  # print the base64 string of the image
print(post_image_to_api(image1))

image_path = 'data/splitted/test/malignant/malignant (150).png'
image2 = image_from_local_path_to_base64(image_path, "PNG")
#print(image2)  # print the base64 string of the image
print(post_image_to_api(image2))
