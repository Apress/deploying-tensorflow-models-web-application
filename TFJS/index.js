let model; 
const class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] //class names

async function loadModel() {
    //function to load the model
    console.log('loading model')
    model = await tf.loadLayersModel('http://127.0.0.1:8000/model_files/model.json') // replace with the correct path 
    console.log( "Model loaded." );
}

function uploadButtonPressed(event)
{   // helper function to display an image  when a user uploads one
    var image = document.getElementById('output');
    image.src =  URL.createObjectURL(event.target.files[0]) 
    image.style.display = "inline";
    document.getElementById("results").style.display = "none";    
}

function predictButtonPressed(){
    //  capture the image and convert into a tensor
   let image  = document.getElementById("output"); 
   let tensor = tf.browser.fromPixels(image)  
		.resizeNearestNeighbor([32, 32])
        .toFloat()
        .expandDims();
    
   // make prediction  
   var predictions = model.predict(tensor);
   const axis = 1 // axis to find maximum value
   const index = predictions.argMax(axis).arraySync()[0]; // index containing the maximum value 
   document.getElementById("results").style.display = "inline"; // display the content
   document.getElementById("results").textContent = "Predicted class is :" + class_names[index]; // display the result
}



