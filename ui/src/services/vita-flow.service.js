import axios from "axios";

const baseUrl = process.env.REACT_APP_VITAFLOW_APIURL;

const uploadImage = file => {
  let data = new FormData();
  data.append("image", file);

  return axios({
    method: "POST",
    url: `${baseUrl}/tie/upload-image`,
    data: data,
    config: {
      headers: {
        "content-type": "multipart/form-data"
      }
    }
  });
};

const processImage = () => {
  return axios(`${baseUrl}/tie/process-image`);
};

const getProcessedImage = fileName => {
  return new Promise((resolve, reject) => {
    processImage().then(
      () => {
        axios(
          `${baseUrl}/tie/get-text-localization?file_name=${fileName}`
        ).then(
          res => {
            resolve(res);
          },
          err => {
            reject(err);
          }
        );
      },
      err => {
        reject(err);
      }
    );
  });
};

const getTextExtracted = () => {
  return axios(`${baseUrl}/tie/get-localized-text`);
};

const vitaFlowService = {
  uploadImage,
  processImage,
  getProcessedImage,
  getTextExtracted
};

export default vitaFlowService;
