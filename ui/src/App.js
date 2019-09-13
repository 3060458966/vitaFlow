import React, { Component } from "react";
import ReactDOM from "react-dom";
import "./App.scss";

import Header from "./components/Header/Header";
import UploadForm from "./components/UploadForm/UploadForm";
import vitaFlowService from "./services/vita-flow.service";

class App extends Component {
  constructor(props) {
    super(props);
    this.textContainerRef = React.createRef();

    this.state = {
      file: null,
      uploaded_file_name: "",
      processed_image: "",
      processed_text: null,
      isUploading: false,
      isFetchingProcessedImage: false,
      isFetchingText: false
    };
  }

  handleInputFileChange = file => {
    this.setState({
      uploaded_file_name: "",
      processed_image: "",
      processed_text: null,
      isUploading: false,
      isFetchingProcessedImage: false,
      isFetchingText: false,
      file: file
    });
  };

  handleUpload = () => {
    const { file } = this.state;
    if (file) {
      this.setState({
        isUploading: true
      });
      vitaFlowService.uploadImage(file).then(
        res => {
          let result = res.data;
          this.setState({
            uploaded_file_name: result.img_file,
            isUploading: false
          });
          this.fetchProcessedImage();
        },
        err => {
          this.setState({
            isUploading: false
          });
        }
      );
    }
  };

  fetchProcessedImage = () => {
    const { uploaded_file_name } = this.state;
    this.setState({
      isFetchingProcessedImage: true
    });
    vitaFlowService.getProcessedImage(uploaded_file_name).then(
      res => {
        let result = res.data;
        this.setState({
          processed_image: result.localised_image,
          isFetchingProcessedImage: false
        });
        this.fetchProcessedText();
      },
      err => {
        this.setState({
          isFetchingProcessedImage: false
        });
      }
    );
  };

  fetchProcessedText = () => {
    this.setState({
      isFetchingText: true
    });
    this.scrollToTextContainer();
    vitaFlowService.getTextExtracted().then(
      res => {
        let result = res.data;
        this.setState({
          processed_text: result,
          isFetchingText: false
        });
        this.scrollToTextContainer();
      },
      err => {
        this.setState({
          isFetchingText: false
        });
      }
    );
  };

  getFormattedString = (obj, type) => {
    const { uploaded_file_name } = this.state;
    let fileName = uploaded_file_name.split(".")[0];
    let data = obj[`${fileName}_${type}`];
    return data.join("\n");
  };

  scrollToTextContainer = () => {
    const element = ReactDOM.findDOMNode(this.textContainerRef.current);
    if (element) {
      element.scrollIntoView({
        behaviour: "smooth"
      });
    }
  };

  handleReset = () => {
    this.setState({
      file: null,
      uploaded_file_name: "",
      processed_image: "",
      processed_text: null,
      isUploading: false,
      isFetchingProcessedImage: false,
      isFetchingText: false
    });
  };

  render() {
    const {
      isUploading,
      isFetchingProcessedImage,
      uploaded_file_name,
      isFetchingText,
      processed_image,
      processed_text,
      file
    } = this.state;
    return (
      <>
        <Header />
        <div className="container-fluid wrapper">
          <div className="form-wrapper">
            <UploadForm
              onFileChange={this.handleInputFileChange}
              onUpload={this.handleUpload}
              onReset={this.handleReset}
            />
          </div>
          <>
            {isUploading || uploaded_file_name ? (
              <div className="image-wrapper">
                <div>
                  <div className="image-title">Uploaded Image</div>
                  <div className="original-image-container">
                    {isUploading ? (
                      <div className="loader-container">
                        <img src="assets/images/loader.svg" alt="loader" />
                        <div className="pt-3 text-secondary">Uploading...</div>
                      </div>
                    ) : uploaded_file_name ? (
                      <img
                        className="image"
                        src={URL.createObjectURL(file)}
                        alt="loader"
                      />
                    ) : null}
                  </div>
                </div>
                <div className="arrow-container">
                  <div className="arrow">&#8594;</div>
                </div>
                <div>
                  <div className="image-title">Processed Image</div>
                  <div className="processed-image-container">
                    {isFetchingProcessedImage ? (
                      <div className="loader-container">
                        <img src="assets/images/loader.svg" alt="loader" />

                        <div className="pt-3 text-secondary">Processing...</div>
                      </div>
                    ) : processed_image ? (
                      <img
                        className="image"
                        src={"data:image/jpg;base64," + processed_image}
                        alt="loader"
                      />
                    ) : null}
                  </div>
                </div>
              </div>
            ) : null}
            {isFetchingText || processed_text ? (
              <div className="text-wrapper" ref={this.textContainerRef}>
                {isFetchingText ? (
                  <div className="loader-container">
                    <img src="assets/images/white-loader.svg" alt="loader" />
                    <div className="pt-3 text-white">Extracting Text...</div>
                  </div>
                ) : processed_text ? (
                  <>
                    <div className="textarea-wrapper">
                      <div className="title">Calamari</div>
                      <textarea
                        value={this.getFormattedString(
                          processed_text,
                          "calamari"
                        )}
                      ></textarea>
                    </div>
                    <img
                      className="image"
                      src={"data:image/jpg;base64," + processed_image}
                      alt="loader"
                    />
                    <div className="textarea-wrapper">
                      <div className="title">Tesseract</div>
                      <textarea
                        value={this.getFormattedString(
                          processed_text,
                          "tesseract"
                        )}
                      ></textarea>
                    </div>
                  </>
                ) : null}
              </div>
            ) : null}
          </>
        </div>
      </>
    );
  }
}

export default App;
