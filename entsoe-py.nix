{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  wheel,
  requests,
  pytz,
  beautifulsoup4,
  pandas,
}:

buildPythonPackage rec {
  pname = "entsoe-py";
  version = "0.8.0";

  src = fetchPypi {
    pname = "entsoe_py";
    inherit version;
    hash = "sha256-r7AS/2JZQZHzd8RowV9ndpx3YrCNVD4EK6BRILSDKyI=";
  };

  # do not run tests
  doCheck = false;

  # specific to buildPythonPackage, see its reference
  pyproject = true;
  build-system = [
    setuptools
    wheel
  ];

  dependencies = [
    requests
    pytz
    beautifulsoup4
    pandas
  ];
}
