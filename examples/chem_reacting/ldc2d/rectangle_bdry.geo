// Gmsh project created on Fri Jun 14 17:33:52 2019
lc = 0.8;
SetFactory("OpenCASCADE");
Point(1) = {-0, -0, -0, lc};
Point(2) = {10, 0, 0, lc};
Point(3) = {10, 1, 0, lc};
Point(4) = {0, 1, -0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(1) = {3, 4, 1, 2};

Plane Surface(1) = {1};

Physical Line(1) = {4};
Physical Line(2) = {2};
Physical Line(3) = {1};
Physical Line(4) = {3};
Physical Surface(5) = {1};

////Use a Distance field to equidistant points in top and bottom boundaries
Field[1] = Distance;
//Field[1].PointsList = {5};
Field[1].CurvesList = {1,3};
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc / 12;
Field[2].SizeMax = lc;
Field[2].DistMin = 0.04;
Field[2].DistMax = 0.15;

Background Field = 2;

// Or use a Box field instead (it's not doing anything - I still need to figure out how to use it)
//Field[1] = Box;
//Field[1].VIn = lc / 12;
//Field[1].VOut = lc;
//Field[1].XMin = 0.0;
//Field[1].XMax = 10.0;
//Field[1].YMin = 0.04;
//Field[1].YMax = 0.15;
//Field[1].Thickness = 0.3;
//
//Field[2] = Box;
//Field[2].VIn = lc / 12;
//Field[2].VOut = lc;
//Field[2].XMin = 0.0;
//Field[2].XMax = 10.0;
//Field[2].YMin = 9.85;
//Field[2].YMax = 9.96;
//Field[2].Thickness = 0.01;
//// Use the minimum
//Field[3] = Min;
//Field[3].FieldsList = {1, 2};
//Background Field = 2;


Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
//Mesh.Algorithm = 5;
