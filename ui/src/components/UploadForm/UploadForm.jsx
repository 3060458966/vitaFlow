import React, { useState } from "react";
import "./UploadForm.scss";

const UploadForm = props => {
  const [file, setFile] = useState(null);
  const { onFileChange, onUpload, onReset } = props;
  const handleFileChange = files => {
    setFile(files[0]);
    onFileChange(files[0]);
  };

  const handleUpload = e => {
    e.preventDefault();
    onUpload();
  };

  const handleReset = e => {
    setFile(null);
    onReset();
  };

  return (
    <div className="upload-view-container">
      <form id="uploadViewForm" onSubmit={handleUpload} onReset={handleReset}>
        <div className="input-group mb-3">
          <div className="custom-file">
            <input
              type="file"
              className="custom-file-input"
              id="browseInput"
              onChange={e => handleFileChange(e.target.files)}
            />
            <label
              className="custom-file-label"
              htmlFor="browseInput"
              aria-describedby="browseInput"
            >
              {file ? file.name : "Choose file"}
            </label>
          </div>
        </div>
        <div className="input-group">
          <button type="submit" className="btn btn-primary mr-2">
            Upload
          </button>
          <button type="reset" className="btn btn-outline-secondary">
            clear
          </button>
        </div>
      </form>
    </div>
  );
};

export default UploadForm;
