//L=4, delta=0.5, h=2.0
SetFactory("OpenCASCADE");
L = 3.0;
delta = 0.2;
h = 1.2;
//+
Point(1) = {-0, 1.0, -0, 0.5};
Point(2) = {L, 1.0, 0, 0.5};
Point(3) = {L, h, 0, 0.5};
Point(4) = {L + 1./delta, h, 0, 0.5};
Point(5) = {L + 1./delta, 1.0, 0, 0.5};
Point(6) = {L + L + 1./delta, 1.0, 0, 0.5};
Point(7) = {L + L + 1./delta, -1.0, 0, 0.5};
Point(8) = {L + 1./delta, -1.0, 0, 0.5};
Point(9) = {L + 1./delta, -h, 0, 0.5};
Point(10) = {L, -h, 0, 0.5};
Point(11) = {L, -1.0, 0, 0.5};
Point(12) = {-0, -1.0, -0, 0.5};
//+
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 1};
//+
Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//+
Plane Surface(1) = {1};
//+
Physical Line("Inflow", 20) = {12};
Physical Line("Outflow", 21) = {6};
Physical Line("Walls", 22) = {1,2,3,4,5,7,8,9,10,11};
//+
Physical Surface("Channel", 23) = {1};
