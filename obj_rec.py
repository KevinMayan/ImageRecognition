from imageai.Detection import ObjectDetection  
 
recognizer = ObjectDetection()  
  
path_model = "./Model/yolo-tiny.h5"  
path_input = "./InputImage/images.jpg"  
path_output = "./OutputImage/newimage.jpg"  
  
recognizer.setModelTypeAsTinyYOLOv3()  

recognizer.setModelPath(path_model)  

recognizer.loadModel()  

recognition = recognizer.detectObjectsFromImage(  
    input_image = path_input,  
    output_image_path = path_output  
    )  
    
for eachItem in recognition:  
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])  
