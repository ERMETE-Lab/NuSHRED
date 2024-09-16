R1 = 1.13;
R2 = 1.83;
R3 = 2.05;

H1 = 0.94;
H2 = 1.13;

delta = 0.2;
R4 = R3 + delta;
H3 = H2 + delta;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {0,  -H2, 0, 1.0};
Point(3) = {R3, -H2, 0, 1.0};
Point(4) = {R3,  H2, 0, 1.0};
Point(5) = {0,   H2, 0, 1.0};

Point(6) = {R1, -H1, 0, 1.0};
Point(7) = {R2, -H1, 0, 1.0};
Point(8) = {R1,  H1, 0, 1.0};
Point(9) = {R2,  H1, 0, 1.0};

Point(10) = {0,  -H3, 0, 1.0};
Point(11) = {R4, -H3, 0, 1.0};
Point(12) = {R4,  H3, 0, 1.0};
Point(13) = {0,   H3, 0, 1.0};

Line(1) = {2, 3};
Line(2) = {3, 4};
Line(3) = {4, 5};
Line(4) = {5, 2};
Line(5) = {6, 7};
Line(6) = {7, 9};
Line(7) = {9, 8};
Line(8) = {8, 6};
Line(9) = {2, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 5};

Curve Loop(1) = {3, 4, 1, 2};
Curve Loop(2) = {8, 5, 6, 7};
Curve Loop(3) = {12, 13, -3, -2, -1, 9, 10, 11};

Plane Surface(1) = {1, 2};
Plane Surface(2) = {3};

Physical Surface("core", 10) = {1};
Physical Surface("reflector", 20) = {2};
Physical Curve("sym", 1) = {4, 13, 9};
Physical Curve("void", 2) = {10, 11, 12};

// Mesh.MeshSizeFactor = .01;

