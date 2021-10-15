all:
	gmsh -2 channel.geo -o channel.msh
	gmsh -2 channel.geo -o channel_hdiv.msh
	gmsh -refine channel_hdiv.msh
	gmsh -refine channel_hdiv.msh
