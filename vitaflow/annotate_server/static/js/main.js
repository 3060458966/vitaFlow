
var selection_xy = [];
var tag_error_raised = false;
var new_zone_created = false;
var init_finished = false;
var current_area = null;
MESSAGE_EMPTY_TAG = "You need to enter a tag here"
RED_COLOR = "#C70039"
GREEN_COLOR = "#80A60E"
var changeStatusMessage = false;

var json_annotations = [];
var list_of_tags = [];
var current_visible_area_id = -1;

var user_id = "";
var image_info = {
                url: "",
				id:"",
				folder: "",
                width: 0,
                height: 0,
				annotations: [],
            };

var lastLoadedWidth  = 0;
var lastLoadedHeight = 0;
var firstWidth = 0;

$(document).ready(function ()
{
	$('#image_to_process').on("load", function()
	{

	    console.log("INFO: " + "on('load', function() ");
		changeStatusMessage = false;

		lastLoadedWidth = $('#image_to_process').get(0).naturalWidth;
		lastLoadedHeight = $('#image_to_process').get(0).naturalHeight;

		image_info.width  = lastLoadedWidth;
		image_info.height = lastLoadedHeight;

		console.log("INFO: " + "image_info.width = "  + image_info.width);
		console.log("INFO: " + "image_info.height = " + image_info.height);

		if (firstWidth == 0)
		{
			firstWidth = $('#image_to_process').get(0).width;
		}

		ratio = lastLoadedWidth / lastLoadedHeight;

		col2_width = 2*$(window).width() / 3;

		// Image is to large
		if (image_info.width < col2_width)
		{
			col2_width = image_info.width;
		}

		image_screen_width = col2_width;
		image_screen_height = image_screen_width/ratio;

		// Image is to high
		if (image_screen_height > $(window).height()*0.75)

		{
			image_screen_height = parseInt($(window).height()*0.75,10);
			image_screen_width = parseInt(image_screen_height*ratio,10);
			col2_width = image_screen_width;

		}

		// alert("New image size = " + image_screen_width + "px , " + image_screen_height + "px");

		remaining_x_space = $(window).width() - col2_width;
		col1_width = 1*remaining_x_space /6;
		col3_width = 4*remaining_x_space /6;
		col4_width = 1*remaining_x_space /6;

		$("#image_to_process").attr({width:image_screen_width+"px"});
		$("#image_to_process").attr({height:image_screen_height+"px"});

		// col1_width = ($(window).width() - $('#image_to_process').get(0).width - 350) /2;
		// col2_width = $('#image_to_process').get(0).width;
		// col3_width = ($(window).width() - $('#image_to_process').get(0).width - 350) /2;

		console.log("INFO: " + "$(window).width() = " + $(window).width());
		console.log("INFO: " + "$(window).height() = " + $(window).height());

		console.log("INFO: " + "col1_width = " + col1_width);
		console.log("INFO: " + "col2_width = " + col2_width);
		console.log("INFO: " + "col3_width = " + col3_width);
		console.log("INFO: " + "col4_width = " + col4_width);

		$("#col1").css("width",col1_width+"px");
		$("#col2").css("width",col2_width+"px");
		$("#col3").css("width",col3_width+"px");
		$("#col4").css("width",col4_width+"px");

		// image_screen_width = 600;
		// image_screen_height = image_screen_width/ratio;
		// Set image dimension
		//$("#image_to_process").attr({width:image_screen_width+"px"});
		//$("#image_to_process").attr({height:image_screen_height+"px"});

		ratio_original_to_screen_x = image_screen_width / lastLoadedWidth;
		ratio_original_to_screen_y = image_screen_height / lastLoadedHeight;

		/*$('#image_to_process').selectAreas(
			{
				allowNudge: false,
				// onChanged:onAreaChanged,
				onDeleted:onAreaDeleted,
			});		*/

		// Loop on tag info
		json_annotations.forEach(function(element) {
			var areaOptions = {
				x: (element.x * ratio_original_to_screen_x),
				y: (element.y * ratio_original_to_screen_y),
				width: (element.width * ratio_original_to_screen_x),
				height: (element.height * ratio_original_to_screen_y),
				tag:element.tag, };

			// We have to convert x, y and size to image in the HTML page
			$('#image_to_process').selectAreas('add', areaOptions);
		});

		// Get list of areas
		var areas = $('#image_to_process').selectAreas('areas');
		if (areas.length>0)
		{
			// Selected area is the first one
			current_area = areas[0];
			// Select first area
			onAreaChanged(null, current_area.id, areas)
			$('#image_to_process').selectAreas('set', current_area.id);
		}
		else
		{
			current_area = null;
		}

	});

	var options = {
		//data: list_of_tags,

		url: "static/resources/list_of_tags.json",
		getValue: "name",

		 template: {
			type: "iconRight",
				fields: {
					iconSrc: "icon"
					}
			},


		list: {
			maxNumberOfElements: 8,

			match: {
				enabled: true
			},

			sort: {
				enabled: true
			},

			onClickEvent: function() {
				isTagInAuthorizedList()
			},

			/*onKeyEnterEvent: function() {
				alert("Enter !");
			},
			onChooseEvent: function() {
				alert("onChooseEvent !");
			},*/

		},
	};

	$('#image_to_process').on("changed", function(event, id, areas)
	{
		// console.log("INFO: " + "index.html on(changed)");

		for (var i = 0; i < areas.length; i++)
		{
			// console.log("INFO: " + "area " + areas[i].id);
			if (areas[i].id == id)
			{
				area = areas[i];
				current_area = area;
			}
		}

		if (area == null)
		{
			console.log("INFO: area is null return");
			current_area = null;
			return;
		}

		// Add the top/left offset of the image position in the page
		var region_tag_length = $("#tag_input").val().length;

		// Set back the area tag if stored
		var stored_tag = area.tag
		if (stored_tag.length >= 3)
		{
			setValidInputTag(stored_tag);
			region_tag_length = area.tag.length;
		}
		else
		{
			// Set en empty tag
			setNotYetValidInputTag("");
			region_tag_length = 0;
		}

		// Do not focus if tag is not empty and at begining or after new zone creation
		if ( (init_finished) && (region_tag_length==0) )
		{
			$("#tag_input").focus();
		}
		else
		{
			$("#tag_input").focusout();
		}
		new_zone_created = false

		// Check region size
		if (isAreaTooSmall(area))
		{
			setStatusAndColor("The region is too small (must be >10px).", RED_COLOR)

			newWidth = area.width;
			newHeight = area.height;

			if (newWidth < 10)
			{
				newWidth = 10;
			}

			if (newHeight < 10)
			{
				newHeight = 10;
			}

			$('#image_to_process').selectAreas('resizeArea', current_area.id, newWidth, newHeight);
		}
		else
		{
			if (changeStatusMessage)
			{
				setStatus("Region has been selected.");
			}
		}

	});

	$("#tag_input").easyAutocomplete(options);

	new_zone_created = false

	$('#add_region').click(function(e) {
		// Sets a random selection
			setTagAndRegion();
		});

	$('#validate_button').click(function(e) {
		// Sets a random selection
			validateTagsAndRegions();
		});

	$('#all_annotations_button').click(function(e) {
	        // Show all annotations
			current_visible_area_id = -1;
			var areas = $('#image_to_process').selectAreas('areas');

			for (var i = 0; i < areas.length; i++)
			{
				$('#image_to_process').selectAreas('setVisibility',  areas[i].id, 1);
			}
	});

	$('#none_annotation_button').click(function(e) {
		    // Hide annotations
			var areas = $('#image_to_process').selectAreas('areas');
			current_visible_area_id = -1;

			for (var i = 0; i < areas.length; i++)
			{
				$('#image_to_process').selectAreas('setVisibility',  areas[i].id, 0);
			}
        });

	$('#one_annotation_button').click(function(e) {

			 // Hide annotations
			var areas = $('#image_to_process').selectAreas('areas');
			current_visible_area_id++;

			/*if (areas.length==0)
			{
			    $('#one_annotation_button').button( "option", "label", "");
			}*/

			if (current_visible_area_id+1 > areas.length)
			{
				current_visible_area_id = 0;
			}

			for (var i = 0; i < areas.length; i++)
			{
				// alert("i " + i);
				// alert("current_visible_area_id " + current_visible_area_id);

				if (i == current_visible_area_id)
				{
					$('#image_to_process').selectAreas('setVisibility',  areas[i].id, 1);
					// $('#one_annotation_button').button( "option", "label", "areas[i].tag");
				}
				else
				{
				    $('#image_to_process').selectAreas('setVisibility',  areas[i].id, 0);
				}
			}

		});

    $('#one_annotation_button_reverse').click(function (e) {

        // Hide annotations
        var areas = $('#image_to_process').selectAreas('areas');
        current_visible_area_id--;

        /*if (areas.length==0)
        {
            $('#one_annotation_button').button( "option", "label", "");
        }*/

        if (current_visible_area_id - 1 < 0) {
            current_visible_area_id = 0;
        }

        for (var n = areas.length; n--;) {
            // alert("i " + i);
            // alert("current_visible_area_id " + current_visible_area_id);

            if (n == current_visible_area_id) {
                $('#image_to_process').selectAreas('setVisibility', areas[n].id, 1);
                // $('#one_annotation_button').button( "option", "label", "areas[i].tag");
            }
            else {
                $('#image_to_process').selectAreas('setVisibility', areas[n].id, 0);
            }
        }

    });



	$("#tag_input").focus(function() {

		if (tag_error_raised)
		{
			tag_error_raised = false
			$("#tag_input").val("");
			$('#tag_input').css({'color':'#000'});
		}
	});

	$("#tag_input").focusout(function() {
		isTagInAuthorizedList()
	});

	$("#tag_input").on('keyup', function (e) {
		if (e.keyCode == 13)
		{
		    // alert("Entering #13");
			var region_tag = $("#tag_input").val();
			if (region_tag.length >= 3)
			{
				setTagAndRegion();
			}
		}
	});

	init_finished = true;
})

function setValidInputTag(_tag)
{
	$("#tag_input").val(_tag);
	isTagInAuthorizedList();
}

function setNotYetValidInputTag(_tag)
{
	$("#tag_input").val(_tag);
	isTagInAuthorizedList();
}

function isTagInAuthorizedList()
{
	var region_tag = $("#tag_input").val();

	list_of_tags = $("#tag_input").getAllItemData();

	if (list_of_tags.indexOf(region_tag)>=0)
	{
		$('#tag_input').css({'color':'#14AEE1'});
		return true;
	}
	else
	{
		$('#tag_input').css({'color':'#000'});
		return false;
	}

}

  function onAreaDeleted(event, id, areas)
  {
	if (current_area !=null)
	{
		if (current_area.id == id)
		{
			current_area = null;
			setNotYetValidInputTag("");
		}
	}
	// if current unset tag @todo later
  }


    function onAreaChanged(event, id, areas)
	{
		// alert(id);

		// console.log("INFO: " + "onAreaChanged() " + event);
		// Find area by id
		for (var i = 0; i < areas.length; i++)
		{
			// console.log("INFO: " + "area " + areas[i].id);
			if (areas[i].id == id)
			{
				area = areas[i];
				current_area = area;
			}
		}

		if (area == null)
		{
			console.log("INFO: area is null return");
			current_area = null;
			return;
		}

		// Add the top/left offset of the image position in the page
		var region_tag_length = $("#tag_input").val().length;

		// Set back the area tag if stored
		var stored_tag = area.tag
		if (stored_tag.length >= 3)
		{
			setValidInputTag(stored_tag);
			region_tag_length = area.tag.length;
		}
		else
		{
			// Set en empty tag
			setNotYetValidInputTag("");
			region_tag_length = 0;
		}

		// Do not focus if tag is not empty and at begining or after new zone creation
		if ( (init_finished) && (region_tag_length==0) )
		{
			$("#tag_input").focus();
		}
		else
		{
			$("#tag_input").focusout();
		}
		new_zone_created = false

		// Check region size
		if (isAreaTooSmall(area))
		{
			setStatusAndColor("The region is too small (must be >10px).", RED_COLOR)

			newWidth = area.width;
			newHeight = area.height;

			if (newWidth < 10)
			{
				newWidth = 10;
			}

			if (newHeight < 10)
			{
				newHeight = 10;
			}

			$('#image_to_process').selectAreas('resizeArea', current_area.id, newWidth, newHeight);
		}
		else
		{
			if (changeStatusMessage)
			{
				setStatus("Region has been selected.");
			}
		}
	}

	function isAreaTooSmall(area)
	{
		if ((area.width<10) || (area.height<10))
		{
			return true;
		}
		else
		{
			return false;
		}

	}

  function px(n) {return Math.round(n) + 'px';	}

  function setStatusAndColor(status_text, color)
  {
	$('#status').css('color', color);
	$("#status").text(status_text);
  }

  // Default color is black
  function setStatus(status_text)
  {
	 setStatusAndColor(status_text, "#000");
  }

  function setTagAndRegion()
  {

	changeStatusMessage = true;

	if (tag_error_raised)
	{
		return false;
	}

	if (current_area == null)
	{
		onAreaChanged(null, current_area.id, areas);
	}

	if (isAreaTooSmall(current_area))
	{
		setStatusAndColor("Tag was not set, the region is too small.", RED_COLOR)
		return false;
	}

	var region_tag = $("#tag_input").val();

	if (region_tag.length < 3)
	{
	    // Display an alert
		tag_error_raised = true
		$("#tag_input").val(MESSAGE_EMPTY_TAG);
		$('#tag_input').css({'color':'#999'});

		return false;
	}

	if (!isTagInAuthorizedList())
	{
		setStatusAndColor("This tag is not in the predefined list.", RED_COLOR);
		return false;
	}

	// Just change the tag, get area, id, ...
	$('#image_to_process').selectAreas('setTag', current_area.id, region_tag);
	setStatus("Tag has been set.");

	// Selection another box
	new_zone_created = true	;

	// Init text
	$("#tag_input").focusout();

	return true;
  }

  function validateTagsAndRegions()
  {

     // Process the list of tags
	 var areas = $('#image_to_process').selectAreas('areas');

	 if (areas.length == 0)
	 {
		setStatusAndColor("Create at least one region with a tag before submitting.", RED_COLOR);
		return false;
	 }

	index_tag = 0;
	ratioX = lastLoadedWidth/$('#image_to_process').width();
	ratioY = lastLoadedHeight/$('#image_to_process').height();
	tmp_annotations = [];

	var error_occurs = false;
	$.each(areas, function (id, area) {
	    var tag_info = {tag:"",x:0,y:0,width:0,height:0};
		tag_info.tag = area.tag;
		tag_info.x = area.x * ratioX;
		tag_info.y = area.y * ratioY;
		tag_info.width = area.width * ratioX;
		tag_info.height = area.height * ratioY;
		// To be checked
		tmp_annotations.push(tag_info);
		// image_info.annotations[index_tag] = tag_info;
		index_tag = index_tag + 1;

		if (area.tag.length<3)
		{
			if (areas.length==1)
			{
				setStatusAndColor("You should add a tag to the region.", RED_COLOR);
			}
			else
			{
				setStatusAndColor("You should add a tag to each region.", RED_COLOR);
			}
			error_occurs = true;
			return false;
		}

	});

	if (error_occurs) return false;

	// image_info.annotations = JSON.stringify(tmp_annotations);
	image_info.annotations = tmp_annotations;
	//alert(image_info.annotations);
	console.log("INFO: annotations : " + image_info.annotations);
	console.log("INFO: annotations : " + JSON.stringify(image_info));

	 $.ajax({
       url : 'inc/validateTagsAndRegions',
       type : 'POST',
	   data : {sendInfo : JSON.stringify(image_info) },
	   cache: false,
	   dataType : "json",
	   success : function(data, status)
	   {
	       // Reload a page but with a message
		   setStatusAndColor("Data has been sent.", GREEN_COLOR);
		   loadImage(); // replace the image
		   getOrCreateUserId(); // Count also points later on

		   // Message to user
		   $("#message").text(data.message);
		   $('#image_to_process').selectAreas('reset');
		   image_info = {
                url: "",
				folder: "",
				id:"",
                width: 0,
                height: 0,
				annotations: [],
            };

	   },
	   error : function(data, status, error)
	   {
			setStatusAndColor("Unable to send data, please retry or check your connection.", RED_COLOR)
       }
    });

  }

  function getCookie(cname) {
    var name = cname + "=";
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for(var i = 0; i <ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "";
}

  function getOrCreateUserId()
  {

    user_id = getCookie("user_id");

	if (user_id != "")
	{
		// The user is known
		$("#message").text("Welcome back !");
    }
	else
	{
		// Create a cookie to identify user
		var date = new Date();
		date.setTime(date.getTime() + (365*24*60*60*1000)); // ms
		expires = "; expires=" + date.toUTCString();
		user_id = getRandomToken();
		document.cookie = "user_id="+ user_id + expires + "; path=/";
		$("#message").text("Welcome !");
    }

  }

  function getRandomToken()
  {
    // E.g. 8 * 32 = 256 bits token
    var randomPool = new Uint8Array(32);

    var crypto = window.crypto || window.msCrypto;

    crypto.getRandomValues(randomPool);
    var hex = '';
    for (var i = 0; i < randomPool.length; ++i) {
        hex += randomPool[i].toString(16);
    }
    // E.g. db18458e2782b2b77e36769c569e263a53885a9944dd0a861e5064eac16f1a
    return hex;
  }

  function loadImage()
  {

	  var goto_url = 'inc/getNewImage';
	  var dataString = JSON.stringify(user_id);
	  console.log("INFO: " + "loadImage()");

      //http://127.0.0.1:5000/011.jpg -> 011.jpg
	  var url_tail = document.URL.split("/").pop();

	  if (url_tail.includes(".")) {
	    console.log("INFO: " + "loadImage() Last URL - ended with " + url_tail);
	    goto_url = goto_url + "/" + url_tail;
	    console.log("INFO: new url is - " + goto_url);
	  }

	  $.ajax({
			type: 'POST',
			url: goto_url,
			data: {user_id: dataString},
			dataType: "json",
			success: function(data)
					 {
					    var dataJson = JSON.parse(data);
						image_info.url = dataJson.url;
						image_info.id = dataJson.id;
						image_info.folder = dataJson.folder;
						image_id   = dataJson.id;
						json_annotations = dataJson.annotations;
						$('#image_id').text(image_id);
						$('#image_to_process').attr("src", image_info.url);
						// debugger;
						// document.getElementById("cropper_btn").href = './cropper/' + image_info.url.split('/').pop();
						// console.log('Update the log')
						// console.log('./cropper/' + image_info.url.split('/').pop());

						/*$('#image_to_process').selectAreas(
						{
							allowNudge: false,
							// onChanged:onAreaChanged,
							onDeleted:onAreaDeleted,
						});*/


					 },
			dataType: 'html'
		});


	}


  function reLoadImage()
  {
	  console.log("INFO: reLoadImage");
	  // document.getElementById("image_to_process").src =  "/static/images/wait.gif";
	  var standard_image_url = document.getElementById("image_to_process").src.split("/").pop();

	  // TODO : Request is supposed to a ajax call.
	  document.getElementById("image_to_process").src =  "/static/data/cropper/" + standard_image_url;
	}


  function openCropper()
  {
  	  console.log("INFO: openCropper");
	  var standard_image_url = document.getElementById("image_to_process").src.split("/").pop();
	  window.open("/cropper/" + standard_image_url, '_blank', 'location=yes,height=570,width=520,scrollbars=yes,status=yes');
	}

/*document.onkeyup = function(e) {
  if (e.which == 77) {
    alert("M key was pressed");
  } else if (e.ctrlKey && e.which == 66) {
    alert("Ctrl + B shortcut combination was pressed");
  } else if (e.ctrlKey && e.altKey && e.which == 89) {
    alert("Ctrl + Alt + Y shortcut combination was pressed");
  } else if (e.ctrlKey && e.altKey && e.shiftKey && e.which == 85) {
    alert("Ctrl + Alt + Shift + U shortcut combination was pressed");
  }
};*/