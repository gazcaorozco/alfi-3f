SetFactory("OpenCASCADE");
//Box(1) = {0, 0, 0, 1, 1, 1};
//Physical Surface(1) = {1};
//Physical Surface(2) = {2};
//Physical Surface(3) = {3};
//Physical Surface(4) = {4};
//Physical Surface(5) = {5};
//Physical Surface(6) = {6};
//Physical Volume(7) = {1};

lc = 0.5;
Point(1) = {-0, -0, -0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, -0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(1) = {3, 4, 1, 2};

Plane Surface(1) = {1};
//Physical Line(1) = {4};
//Physical Line(2) = {2};
//Physical Line(3) = {1};
//Physical Line(4) = {3};
//Physical Surface(5) = {1};

//Use a Distance field to equidistant points in top and bottom boundaries
Field[1] = Distance;
Field[1].CurvesList = {2,4};
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc / 2; //3.5
Field[2].SizeMax = lc;
Field[2].DistMin = 0.1; //0.1
Field[2].DistMax = 0.3; //0.3

Background Field = 2;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
//Mesh.Algorithm = 5;

//Now make 3D by extrusion.
newEntities[] =
Extrude { 0,0,1 }
{
	Surface{1};
	Layers{4};
//	Recombine;
};

Physical Surface(1) = {1}; //back
Physical Surface(2) = {newEntities[0]}; //front
Physical Surface(3) = {newEntities[2]}; //bottom
Physical Surface(4) = {newEntities[3]}; //right
Physical Surface(5) = {newEntities[4]};  //top
Physical Surface(6) = {newEntities[5]}; //left
Physical Volume(100) = {newEntities[1]};
