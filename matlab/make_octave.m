function make_octave
  mkoctfile --mex splatt_load.c -I../include -L../build/Linux-x86_64/lib ...
      -lsplatt -lgomp -lm
  mkoctfile --mex  splatt_cpd.c -I../include -L../build/Linux-x86_64/lib ...
      -lsplatt -lgomp -lm
  mkoctfile --mex  splatt_mttkrp.c -I../include -L../build/Linux-x86_64/lib ...
      -lsplatt -lgomp -lm

% TODO: How to handle other operating systems?
