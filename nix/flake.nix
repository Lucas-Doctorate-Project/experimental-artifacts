{
  description = "Experimental artifacts and campaign runner";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    batsim.url = "github:Lucas-Doctorate-Project/batsim";
    batsched.url = "github:Lucas-Doctorate-Project/batsched";
    evalys.url = "github:Lucas-Doctorate-Project/evalys";
  };

  outputs = inputs@{ flake-parts, batsim, batsched, evalys, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
      let
        batsim-pkg = inputs.batsim.packages.${system}.default;
        batsched-pkg = inputs.batsched.packages.${system}.default;
        evalys-pkg = inputs.evalys.packages.${system}.evalys;

        python = pkgs.python3.override {
          self = python;
          packageOverrides = pyfinal: pyprev: {
            entsoe-py = pyfinal.callPackage ./entsoe-py.nix { };
          };
        };

        evalysPython = evalys-pkg.pythonModule;
        evalysPythonEnv = evalysPython.withPackages (ps: [
          ps.pandas
          ps.matplotlib
          evalys-pkg
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            (python.withPackages (python-pkgs: [
              python-pkgs.pandas
              python-pkgs.matplotlib
              python-pkgs.seaborn
              python-pkgs.entsoe-py
              python-pkgs.notebook
              python-pkgs.jupyterlab
              python-pkgs.ipykernel
            ]))
            pkgs.go
            batsim-pkg
            batsched-pkg
            evalysPythonEnv
          ];

          shellHook = "unset GOPATH";
        };
      };
    };
}
