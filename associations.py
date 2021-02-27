import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
from matplotlib import rcParams
rcParams['figure.dpi'] = 150
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set_style("darkgrid")

import ipywidgets as widgets
from ipywidgets import interactive

from IPython.core.display import display, HTML

import pandas as pd
import random

from association import get_df_association
from variable_types import get_df_var_types


class associations:
    
    var_color = {"BOOL":"#c03d3e", "CAT":"#3a923a", "NUM":"#337ab7"}
    
    
    def __init__(self, df):
        
        self.df = df
        self.var_type = get_df_var_types(self.df)
        self.association_df = get_df_association(self.df, self.var_type)
        
        self.num_var = sorted([k for k, v in self.var_type.items()  if (v=="NUM")])
        self.cat_var = sorted([k for k, v in self.var_type.items()  if (v=="CAT")])
        self.bool_var = sorted([k for k, v in self.var_type.items()  if (v=="BOOL")])

        
    def get_association_table(self):
        return self.association_df
    
    
    def get_variable_association(self, var, limit=5, num_num="pearson", num_cat="correlation_ratio", cat_cat="theils"):
        
        if self.check_metric(num_num, num_cat, cat_cat):
            return ("Please provide a valid metric name ({})".format(self.association_df["metric"].unique()))
        
        df = self.association_df
        df = df[df["metric"].isin([num_num, num_cat, cat_cat])]
        df = df[df["col_a"] != df["col_b"]].sort_values("association", ascending=False)
        
        
        if (self.var_type[var] in ["BOOL", "CAT"]):
            
            ## numeriacal assocciation
            num_assn = df[(df["col_a"] == var) & (df["type_"] == "CAT-NUM")]
            num_assn = num_assn.drop(["col_a", "type_"], axis=1).rename(columns={"col_b":"Variable"}).reset_index(drop=True)
            num_assn = num_assn[["Variable", "association"]].head(limit)

            ## parent association
            parent_assn = df[(df["col_a"] == var) & (df["type_"] == "CAT-CAT")]
            parent_assn = parent_assn.drop(["col_a", "type_"], axis=1).rename(columns={"col_b":"Variable"}).reset_index(drop=True)
            parent_assn = parent_assn[["Variable", "association"]].head(limit)

            ## child association
            child_assn = df[(df["col_b"] == var) & (df["type_"] == "CAT-CAT")]
            child_assn = child_assn.drop(["col_b", "type_"], axis=1).rename(columns={"col_a":"Variable"}).reset_index(drop=True)
            child_assn = child_assn[["Variable", "association"]].head(limit)
        
            display_side_by_side([num_assn, parent_assn, child_assn], ['Numerial Association', 'Parent Association', 'Child Association'])
            
            
        elif (self.var_type[var] in ["NUM"]):
            
            df = df[df["col_a"] == var].drop("col_a", axis=1).rename(columns={"col_b":"Variable"})[["Variable", "association", "type_"]]
            ## positive and negative association
            num_assn = df[df["type_"] == "NUM-NUM"].drop("type_", axis=1)
            pos_num_assn = num_assn[num_assn["association"] > 0].head(limit).reset_index(drop=True)
            neg_num_assn = num_assn[num_assn["association"] < 0].sort_values("association").head(limit).reset_index(drop=True)

            ## categorical association
            cat_assn = df[df["type_"] == "NUM-CAT"].drop("type_", axis=1)
            cat_assn = cat_assn[cat_assn["association"] > 0.01].head(limit).reset_index(drop=True)
            
            display_side_by_side([pos_num_assn, neg_num_assn, cat_assn], ['Positive Association', 'Negative Association', 'Categorical'])
        
        else:
            print ("Please provide categorical or numerical attribute")
            
            
            
    def heatmap_plot(self, num_num="pearson", num_cat="correlation_ratio", cat_cat="theils"):
        
        if self.check_metric(num_num, num_cat, cat_cat):
            return ("Please provide a valid metric name ({})".format(self.association_df["metric"].unique()))
        
        rcParams['figure.figsize'] = (8, 5)

        def f(min_associaion):
            df = self.association_df
            df = df[df["metric"].isin([num_num, num_cat, cat_cat])]
            heatmap_df = df[abs(df["association"]) > min_associaion]
            heatmap_df = heatmap_df[abs(heatmap_df["association"]) < 1]
            heatmap_df = heatmap_df.pivot_table(index="col_a", columns="col_b", values="association")
            heatmap_df = heatmap_df.reindex([i for i in  self.num_var+self.cat_var+self.bool_var if i in heatmap_df.columns])
            heatmap_df = heatmap_df[[i for i in  self.num_var+self.cat_var+self.bool_var if i in heatmap_df.columns]]
            
            ax = sns.heatmap(heatmap_df,
                             center=0, vmin=0, vmax=1, linewidths=1.5,
                             cmap="RdBu", annot=True, annot_kws={'size':8}, xticklabels=True, yticklabels=True)
            for ticklabel in ax.get_xticklabels():
                ticklabel.set_color(self.var_color[self.var_type[ticklabel.get_text()]])

            for ticklabel in ax.get_yticklabels():
                ticklabel.set_color(self.var_color[self.var_type[ticklabel.get_text()]])
            
            plt.title("Association", fontsize=15, weight='bold')
            plt.xlabel('Target', fontsize=0, weight='bold')
            plt.xticks(fontsize=10, rotation=90)

            plt.ylabel('Child', fontsize=0, weight='bold')
            plt.yticks(fontsize=10, rotation=0)
            plt.show()
            

            
        w1 = widgets.FloatSlider(value=0, min=0, max=1.0, step=0.01, description='Min. Association:',
                                 layout=widgets.Layout(width='400px', height='40px', border='1px solid black'),
                                 style={'description_width': 'initial'})
        w1.style.handle_color = 'lightblue'
        interactive_plot = interactive(f, min_associaion=w1)
        
        return interactive_plot
    
    
    def network_plot(self, seed=4, fig_size=(8, 5), size_node=2000, size_edge=100, size_label=10, limit_label=0.4, num_num="pearson", num_cat="correlation_ratio", cat_cat="theils"):
        
        if self.check_metric(num_num, num_cat, cat_cat):
            return ("Please provide a valid metric name ({})".format(self.association_df["metric"].unique()))
        
        rcParams['figure.figsize'] = (8, 5)
        
        
        def f(limit, size_node, size_edge, size_label, seed):
            
            df = self.association_df
            df = df[df["metric"].isin([num_num, num_cat, cat_cat])]
            df["association"] = abs(df["association"])

            ## filter
            df = df[df["association"] >= limit]
            df = df[df["col_a"] != df["col_b"]]

            ## nodes sizes
            size_df = df.copy()
            size_df["association"] = size_df["association"].apply(lambda x: (10*x)**2)
            size_df = size_df.groupby("col_a", as_index=False)["association"].sum().rename(columns={"association":"size"})
            size_df["size"] = size_df["size"]/size_df["size"].max() ## normalizing (max size=1)

            ## creating graph
            G = nx.from_pandas_edgelist(df,
                                        source='col_a', target='col_b',
                                        edge_attr=["association", "type_"],
                                        create_using=nx.DiGraph())

            ## size and color attribute of nodes
            for i in list(G.nodes()):
                G.nodes[i]['size'] = size_df[size_df['col_a']==i]['size'].values[0]

                if (self.var_type[i] == "BOOL"):
                    G.nodes[i]['color'] = self.var_color["BOOL"]
                elif (self.var_type[i] == "CAT"):
                    G.nodes[i]['color'] = self.var_color["CAT"]
                elif (self.var_type[i] == "NUM"):
                    G.nodes[i]['color'] = self.var_color["NUM"]
                else:
                    G.nodes[i]['color'] = "blue"

            ## color, size, width
            node_color = [nx.get_node_attributes(G, 'color')[v] for v in G] 
            node_size = [size_node*nx.get_node_attributes(G, 'size')[v] for v in G]  
            edge_width = [size_edge*G[u][v]['association']**2 for u, v in G.edges()]


            ## drawing
            ## 1. size of the figure 
            rcParams['figure.figsize'] = fig_size

            ## 2. layout
            pos = nx.spring_layout(G, iterations=50, seed=seed)

            ## 3. nodes
            nx.draw_networkx_nodes(G, pos,
                                   node_color = node_color, node_size = node_size,
                                   edgecolors="black", linewidths=0.5, node_shape = "o", alpha=1, font_size=10)

            ## 4. node labels
            for node, (x, y) in pos.items():
                text(x, y, node, fontsize=size_label*(nx.get_node_attributes(G, 'size')[node])**0.5, ha='center', va='center')

            ## 5. edges
            nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.15, style='solid', edge_color="grey", arrows=False)

            ## 6. edge labels
            edge_labels =dict([((u, v), d['association']) for u, v, d in G.edges(data=True) if (d["association"]>limit_label)])
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            plt.axis('off') 
            plt.tight_layout()
            plt.margins(0.1)
            
            ## legends
            colors = [self.var_color["BOOL"], self.var_color["CAT"], self.var_color["NUM"]]
            texts = ["Bool", "Cat", "Num"]
            patches = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i],\
                                 label="{:s}".format(texts[i]) )[0]  for i in range(len(texts))]

            plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

            

        w1 = widgets.FloatSlider(value=0.05, min=0, max=1.0, step=0.01, description='Min. Association:',
                                 layout=widgets.Layout(width='400px', height='40px', border='1px solid black'),
                                 style={'description_width': 'initial'})
        w2 = widgets.FloatSlider(value=2000, min=0, max=10000, step=100, description='Node Size:',
                                 layout=widgets.Layout(width='400px', height='40px', border='1px solid black'),
                                 style={'description_width': 'initial'})
        w3 = widgets.FloatSlider(value=100, min=0, max=1000, step=10, description='Edge Size:',
                                 layout=widgets.Layout(width='400px', height='40px', border='1px solid black'),
                                 style={'description_width': 'initial'})
        w4 = widgets.FloatSlider(value=10, min=0, max=40, step=1, description='Label Size:',
                                 layout=widgets.Layout(width='400px', height='40px', border='1px solid black'),
                                 style={'description_width': 'initial'})
        w5 = widgets.IntSlider(value=10, min=0, max=40, step=1, description='Seed:',
                                 layout=widgets.Layout(width='400px', height='40px', border='1px solid black'),
                                 style={'description_width': 'initial'}, )
               
        def make_boxes():
            vbox1 = widgets.VBox([w1, w2, w3], )
            vbox2 = widgets.VBox([w4, w5])
            return vbox1, vbox2

        vbox1, vbox2 = make_boxes()

        box_layout = widgets.Layout( border='solid 1px red', margin='0px 10px 10px 0px', padding='5px 5px 5px 5px')

        vbox1.layout = box_layout
        vbox2.layout = box_layout

        
        ui = widgets.HBox([vbox1, vbox2])
        interactive_plot = widgets.interactive_output(f, {"limit":w1, "size_node":w2, "size_edge":w3, "size_label":w4, "seed":w5})
        display(ui, interactive_plot)

    
    def check_metric(self, num_num, num_cat, cat_cat):
        for i in [num_num, num_cat, cat_cat]:
            if i not in self.association_df["metric"].unique():
                return True
        return False

    


def display_side_by_side(dfs:list, captions:list):
    
    output = ""
    combined = dict(zip(captions, dfs))
    styles = [dict(selector="caption",
                   props=[("text-align", "center"), ("font-size", "120%"), ("color", 'Red'),
                          ("font-weight", "bold"), ("border", "1px solid lightgrey")])]
    
    styles.append(dict(selector='th', props=[('text-align', 'center')]))
    
    for caption, df in combined.items():
        output += df.style.bar(subset=['association'], vmax=1, vmin=-1, align='mid', color=['#d65f5f', '#5fba7d']).format({"association": lambda x: "{:.2f}".format(x)}).set_table_attributes("style='display:inline'").set_caption(caption).set_table_styles(styles)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))
    
    
