import numpy as np
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

def make_sphere(center, radius, color, res = 8, opacity=1.0): 
    
    # Set up 100 points. First, do angles
    theta = np.linspace(0,2*np.pi,res)
    phi = np.linspace(0,np.pi,res)
    
    # Set up coordinates for points on the sphere
    x0 = center[0] + radius * np.outer(np.cos(theta),np.sin(phi)).flatten()
    y0 = center[1] + radius * np.outer(np.sin(theta),np.sin(phi)).flatten()
    z0 = center[2] + radius * np.outer(np.ones(res),np.cos(phi)).flatten()
    
    trace = go.Mesh3d(x=x0, y=y0, z=z0, color=color, hoverinfo='skip', alphahull=0, opacity=opacity)
    trace.update(showscale=False)

    return trace

def make_graph_meshes(graph, opacity=1.0):
    meshes = []
    for node in graph.ndata:
        pos = node.coord
        radius = node.get_radius()
        color = node.get_color()
        meshes.append(make_sphere(pos, radius, color, opacity=opacity))
    for src, dst in graph.edges:
        xs = [graph.ndata.coord[src][0], graph.ndata.coord[dst][0]]
        ys = [graph.ndata.coord[src][1], graph.ndata.coord[dst][1]]
        zs = [graph.ndata.coord[src][2], graph.ndata.coord[dst][2]]
        meshes.append(go.Scatter3d(x=xs, y=ys, z=zs, hoverinfo='skip', showlegend=False, line={"color": "black"}))
    return meshes

def plot_meshes(meshes):
    axis_params = dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, titlefont=dict(color='white'), showspikes=False)
    layout = dict(scene=dict(xaxis=axis_params, yaxis=axis_params, zaxis=axis_params))
    fig = go.Figure(data=meshes, layout=layout)
    iplot(fig)

def plot_graph(graph):
    plot_meshes(make_graph_meshes(graph))

def plot_graphs(graphs):
    meshes = []
    for graph in graphs:
        meshes += make_graph_meshes(graph)
    plot_meshes(meshes)