function [A,B,C,D] = RondeParametricBeamModel()
A1 = [0 1 ;-513.4 -9.5];
A2 = [0 1 ; -1447.3 -10.7];
A3 = [0 1 ; -43954.1 -8.3];
A4 = [0 1 ; -375432.6 -19.2];
A5 = [0 1; -1270020.5 -9.1];
A6 = [0 1; -3233164.0 -36.1];
A7 = [0 1 ; -7625745.1 -12];
A = blkdiag(A1,A2,A3,A4,A5,A6,A7);
B = [0      0       0;
    -35.7   -47.8   -54;
    0       0       0;
    -48.7   -6.7    35.6;
    0       0       0;
    -44.5   60.5    -36.0;
    0       0       0;
    -8.4    6.9     -3.5;
    0       0       0;
    -21.2   62      -24.1;
    0       0       0;
    -9.6    -0.2    8.3;
    0       0       0;
    37.4    36.6    38.9];
C = [-51.4 0 -152.4 0 -73.9 0 -11 0 -5.3 0 -34.9 0 -50.4 0;
    -78.1 0 -26.6 0 83.5 0 6.2 0 67.5 0 8.1 0 -67.9 0;
    -88.7 0 111.4 0 -42.7 0 -7.1 0 -43.1 0 75.4 0 -78.1 0];
D = [0.0042 0.0002 0.0002;
    -0.0003 0.0043 0.0010;
    -0.0001 0.0004 0.0049];
return;
end

