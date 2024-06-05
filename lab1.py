import heapq
import os
import re
import tkinter as tk
from math import inf, isinf
from random import choice
from tkinter import ttk
from tkinter.filedialog import askopenfile
from tkinter.messagebox import showerror, showinfo
from typing import Dict, Generator, List, Optional, cast

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def all_simple_paths_graph(G: nx.Graph, source: str, targets: str) -> Generator[List[str], None, None]:
    cutoff = len(G) - 1
    visited = dict.fromkeys([source])
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child == targets:
                yield list(visited) + [child]
            visited[child] = None
            if {targets} - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            for target in ({targets} & (set(children) | {child})) - set(visited.keys()):
                yield list(visited) + [target]
            stack.pop()
            visited.popitem()


class SideFrame(tk.Frame):
    def setup(self) -> None:
        self.input_notebook = ttk.Notebook(self, width=50)
        self.input_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.output = tk.Text(self, width=40, height=10, state=tk.DISABLED)
        self.output.pack(side=tk.BOTTOM, fill=tk.X, expand=True, pady=(10, 0))
        self.setup_bridge_words()
        self.setup_generate_text()
        self.setup_shortest_path()
        self.setup_random_traversal()
        self.deactivate()

    def setup_bridge_words(self) -> None:
        self.bridge_words_frame = ttk.Frame(self.input_notebook)
        self.input_notebook.add(self.bridge_words_frame, text="Bridge Words")
        self.bridge_words_input1 = ttk.Entry(self.bridge_words_frame)
        self.bridge_words_input1.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.bridge_words_input2 = ttk.Entry(self.bridge_words_frame)
        self.bridge_words_input2.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.bridge_words_button = ttk.Button(
            self.bridge_words_frame,
            text="Find Bridge Words",
            command=self.query_bridge_words_callback,
        )
        self.bridge_words_button.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def setup_random_traversal(self) -> None:
        self.random_traversal_frame = ttk.Frame(self.input_notebook)
        self.input_notebook.add(self.random_traversal_frame, text="Random Walk")
        self.random_traversal_button = ttk.Button(
            self.random_traversal_frame,
            text="Start",
            command=self.random_traversal_callback,
        )
        self.random_traversal_button.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def setup_generate_text(self) -> None:
        self.generate_text_frame = ttk.Frame(self.input_notebook)
        self.input_notebook.add(self.generate_text_frame, text="Generate Text")
        self.generate_text_input = ttk.Entry(self.generate_text_frame)
        self.generate_text_input.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.generate_text_button = ttk.Button(
            self.generate_text_frame,
            text="Generate",
            command=self.generate_text_callback,
        )
        self.generate_text_button.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def setup_shortest_path(self) -> None:
        self.shortest_path_frame = ttk.Frame(self.input_notebook)
        self.input_notebook.add(self.shortest_path_frame, text="Shortest Path")
        self.shortest_path_input1 = ttk.Entry(self.shortest_path_frame)
        self.shortest_path_input1.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.shortest_path_input2 = ttk.Entry(self.shortest_path_frame)
        self.shortest_path_input2.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.shortest_path_button = ttk.Button(
            self.shortest_path_frame,
            text="Find Path",
            command=self.shortest_path_callback,
        )
        self.shortest_path_button.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def set_output(self, text: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state=tk.DISABLED)

    def query_bridge_words_callback(self) -> None:
        master = cast(MainWindow, self.master)
        word1 = self.bridge_words_input1.get()
        word2 = self.bridge_words_input2.get()
        if word1 not in master.graph.nodes or word2 not in master.graph.nodes:
            showerror("Error", "No word1 or word2 in the graph!")
            return
        self.set_output(master.query_bridge_words(word1, word2))

    def random_traversal_callback(self) -> None:
        master = cast(MainWindow, self.master)
        self.set_output(master.random_walk())

    def generate_text_callback(self) -> None:
        master = cast(MainWindow, self.master)
        input_text = self.generate_text_input.get()
        self.set_output(master.generate_new_text(input_text))

    def shortest_path_callback(self) -> None:
        master = cast(MainWindow, self.master)
        word1 = self.shortest_path_input1.get()
        word2 = self.shortest_path_input2.get()
        if word1 not in master.graph.nodes or word2 not in master.graph.nodes:
            showerror("Error", "No word1 or word2 in the graph!")
            return
        if path := master.calc_shortest_path(word1, word2):
            self.set_output(f"Shortest path: {' -> '.join(path.split())}\n")
        else:
            showinfo("Info", "No path found between the two words!")

    def activate(self) -> None:
        self.bridge_words_input1.config(state=tk.NORMAL)
        self.bridge_words_input2.config(state=tk.NORMAL)
        self.bridge_words_button.config(state=tk.NORMAL)
        self.generate_text_input.config(state=tk.NORMAL)
        self.generate_text_button.config(state=tk.NORMAL)
        self.random_traversal_button.config(state=tk.NORMAL)
        self.shortest_path_input1.config(state=tk.NORMAL)
        self.shortest_path_input2.config(state=tk.NORMAL)
        self.shortest_path_button.config(state=tk.NORMAL)

    def deactivate(self) -> None:
        self.bridge_words_input1.config(state=tk.DISABLED)
        self.bridge_words_input2.config(state=tk.DISABLED)
        self.bridge_words_button.config(state=tk.DISABLED)
        self.generate_text_input.config(state=tk.DISABLED)
        self.generate_text_button.config(state=tk.DISABLED)
        self.random_traversal_button.config(state=tk.DISABLED)
        self.shortest_path_input1.config(state=tk.DISABLED)
        self.shortest_path_input2.config(state=tk.DISABLED)
        self.shortest_path_button.config(state=tk.DISABLED)
        self.set_output('')


class MainWindow(ttk.Frame):
    def setup(self) -> None:
        self.load_button = ttk.Button(self, text="Load", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=(250, 250), pady=(250, 250))
        self.graph = nx.DiGraph()
        self.graph_layout = {}
        self.side_menu = SideFrame(self, width=20)
        self.side_menu.setup()
        self.side_menu.pack(side=tk.RIGHT, fill=tk.Y)

    def show_directed_graph(self) -> None:
        f = plt.figure(figsize=(7, 7))
        pos = nx.spring_layout(self.graph, iterations=256)
        nx.draw(self.graph, with_labels=True, pos=pos)
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)

        self.load_button.pack_forget()
        self.graph_figure = f
        self.graph_layout = pos
        self.graph_canvas = FigureCanvasTkAgg(f, master=self)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(side=tk.LEFT)

    def query_bridge_words(self, word1: str, word2: str) -> str:
        """get bridge words

        :param word1: input word1
        :type word1: str
        :param word2: input word2
        :type word2: str
        :return: bridge words split by ' '
        :rtype: str
        """
        if word1 not in self.graph.nodes or word2 not in self.graph.nodes:
            return ""
        paths = all_simple_paths_graph(self.graph, word1, word2)
        words = set()
        for path in paths:
            words.update(path[1:-1])
        return " ".join(words)

    def generate_new_text(self, input_text: str) -> str:
        """generate new text based on the bridge word

        :param word1: input word1
        :type word1: str
        :param word2: input word2
        :type word2: str
        :return: new text
        :rtype: str
        """
        words = re.split(r"[^A-Za-z]+", input_text.lower())
        if len(words) < 2:
            return input_text
        new_text = [words[0]]
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            if bridge_words := self.query_bridge_words(word1, word2):
                new_text.append(choice(bridge_words.split()))
            new_text.append(word2)
        return " ".join(new_text)

    def calc_shortest_path(self, word1: str, word2: str) -> str:
        """get shortest path

        :param word1: input word1
        :type word1: str
        :param word2: input word2
        :type word2: str
        :return: path split by ' '
        :rtype: str
        """
        if word1 not in self.graph.nodes or word2 not in self.graph.nodes:
            return ""

        distances = {node: inf for node in self.graph.nodes}
        previous_nodes: Dict[str, Optional[str]] = {node: None for node in self.graph.nodes}
        distances[word1] = 0
        priority_queue = [(0, word1)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node == word2:
                break

            if current_distance > distances[current_node]:
                continue

            for neighbor, attributes in self.graph[current_node].items():
                weight = attributes.get('weight', 1)
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        path = []
        current_node = word2
        while prev := previous_nodes[current_node]:
            path.insert(0, current_node)
            current_node = prev
        if path:
            path.insert(0, current_node)

        if isinf(distances[word2]):
            return ""
        self.highlight_path(path)
        return ' '.join(path)

    def random_walk(self) -> str:
        """random walk

        :return: path split by ' '
        :rtype: str
        """
        used = set()
        start = choice(list(self.graph.nodes))
        result = [start]
        while True:
            adj = list(self.graph.adj[start])
            if not adj:
                break
            nxt = choice(adj)
            result.append(nxt)
            if (start, nxt) in used:
                break
            used.add((start, nxt))
            start = nxt
        return " ".join(result)

    def highlight_path(self, path: List[str]) -> None:
        self.graph_figure.clear()
        pos = self.graph_layout
        nx.draw(self.graph, with_labels=True, pos=pos)
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color='red')
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='red', width=2)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=nx.get_edge_attributes(self.graph, 'weight'))
        self.graph_canvas.draw()

    def load_file(self) -> None:
        file = askopenfile(
            "rb",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Select an input file",
            initialdir=os.getcwd(),
        )
        if file is None:
            return
        with file:
            text: str = file.read().decode()
        last = None
        for t in re.split(r"[^A-Za-z]+", text.lower()):
            if not t:
                continue
            self.graph.add_node(t)
            if last is not None:
                if (last, t) not in self.graph.edges:
                    self.graph.add_edge(last, t, weight=1)
                else:
                    self.graph.edges[last, t]["weight"] += 1
            last = t
        self.side_menu.activate()
        self.show_directed_graph()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Lab1")
    root.geometry("1080x720")
    root.resizable(False, False)
    main = MainWindow(root, width=1080, height=720)
    main.setup()
    main.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    root.mainloop()
