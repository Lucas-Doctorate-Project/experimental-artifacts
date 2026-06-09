{
  description = "Description for the project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
      let
        python = pkgs.python3.override {
          self = python;
          packageOverrides = pyfinal: pyprev: {
            entsoe-py = pyfinal.callPackage ./entsoe-py.nix { };
          };
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            (python.withPackages (python-pkgs: [
              python-pkgs.pandas
              python-pkgs.matplotlib
              python-pkgs.entsoe-py
            ]))
          ];
        };
      };
    };
}
