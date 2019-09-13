import React, { Component } from "react";
import "./Header.scss";

class Header extends Component {
  render() {
    return (
      <nav className="navbar navbar-white bg-white app-navbar">
        <img
          src="/assets/icons/vitaflow-logo.png"
          height="40"
          className="d-inline-block align-top"
          alt="logo"
        />
      </nav>
    );
  }
}

export default Header;
