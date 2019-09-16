
function format_row(obj){

// cropper - is optional feature
var cropper_url = obj.hasOwnProperty('cropper_url')? obj.cropper_url : "static/images/NoImage.svg";
var binarisation_url = obj.hasOwnProperty('binarisation_url')? obj.binarisation_url : "static/images/NoImage.svg";

//if ()

html = `<tr>` +
    `<td><p> Input File: <\p><p><a href=` + obj.file + `>link text` + obj.file + `</a></p>` +
    `<img src=/` + obj.url + ` style="height:300px;">` +
    `</td>` +
    `<td><p> Cropper File:</p>` +
    `<img src=/` + cropper_url + ` style="height:300px;">` +
    `<td>` +
    `<td><p> Binarisation File:</p>` +
    `<img src=/` + binarisation_url + ` style="height:300px;">` +
    `<td>` +
    `</tr>`;

return html;
}

var start = 0;
var number_of_images_to_load = 5;


function load_prev_page_item(){

var json_data = {};
var html_data = "";

start = Math.max(0, start - number_of_images_to_load);

$.ajax({
    type: 'GET',
    url: '/summary/' + start + '/' + start + number_of_images_to_load,
    success: function (json_data) {
    // On success code
    var receipt_images = json_data.receipt_images;
    // receipt_images
    for (var key in receipt_images) {
       if (receipt_images.hasOwnProperty(key)) {
          var obj = receipt_images[key];
             html_data = html_data + format_row(obj);
          }
       }
    // receipt_images - output
    document.getElementById("show_image_data").innerHTML = html_data;
    console.log("Loading images from " + start + ' till ' + (start + number_of_images_to_load) );

        }

    })


// end of function
}


function load_next_page_item(){

start = start + number_of_images_to_load;

images_ajax_call(start, start + number_of_images_to_load);
// end of function
}



function images_ajax_call(start, end){

console.log('start is ' + start)
console.log('end is ' + end)

var json_data = {};
var html_data = "";

$.ajax({
    type: 'GET',
    url: '/summary/' + start + '/' + end,
    success: function (json_data) {
    // On success code
    var receipt_images = json_data.receipt_images;
    // receipt_images
    for (var key in receipt_images) {
       if (receipt_images.hasOwnProperty(key)) {
          var obj = receipt_images[key];
             html_data = html_data + format_row(obj);
          }
       }
    // receipt_images - output
    document.getElementById("show_image_data").innerHTML = html_data;
    console.log("Loading images from " + start + ' till ' + end);
        }

    })


// end of function
}

//test()