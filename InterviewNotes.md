# Interview Notes
Interview notes and common questions for multiple topics. Useful to review, maybe not to learn stuff from scratch.
# Topic: Big O

- How long does something take? What is it dependant on?
	- Big O - Worst case
	- Big Omega - Best Case
	- Big Theta - Average case
- Big O and Big Theta [worst and expected] are usually the same - we generally deal with these two
- Space complexity matters too: Recursion generally creates more calls on the stack -> more space occupied
- Drop constants and non-dominant terms
- Common computations
	- Sequential operations => Add
	- Nested operations => Multiply
	- Dividing the size of the operand by 2 => log(n)
	- When recursive functions make multiple calls => 2^N
	- Straight recursion, on the other hand - generally O(n)
	- When arrays are different and being iterated nested-ly => O(ab) - this can happen
	- When you have multiple variables [length of a string, length of an array] -> use a,b,c,x - don't use N to avoid mixups
	
Solving a problem:
- Listen carefully (details - sorted array, run repeatedly => do we want to cache data?)
- Draw it out - good for showing your thought process
- Brute force
- Refactor brute force
	- Look for unused information
	- Use a fresh example
	- Time vs space
	- Precompute data
	- Hash table
	- BUD - Bottlenecks, Unnecessary, Duplicated work
- Walk them through it
- Modularise code
- Sanity checks

Remember:
- nPr is the <b>number of permutations</b>
- nCr is the <b>number of combinations</b>

If they ask you only for counts, just calculate them. Don't actually permute. This saves a bunch of runtime

<b>When stuck:</b>
- Put something down on paper, and observe how your brain lets you work it out
- Simplify the problem and generalise the solution
- Brainstorm possible DSs

<a href="https://stackoverflow.com/questions/29910312/algorithm-to-get-all-the-combinations-of-size-n-from-an-array-java">All combinations of a list of elements</a>:

Basics:
- Binary, Hex, Decimal interconversion. Binary or Hex to decimal, vice versa
- DO NOT GIVE UP

# Topic: Data Structures

## Arrays and Strings

Hash Tables:
- <b>Can solve most problems really well</b>
- O(1) in Python and Java if used correctly
- O(1) lookup for 0 collisions, O(N) for N keys (with lots of collisions) -> Linked List
- O(log N) lookup -> Implemented with balanced BST

ArrayLists: 
O(1) lookup
O(N) expansion

StringBuilder
No O(x) cost to copy strings over when appending multiple strings to each other

StringBuilder s = new StringBuilder();
s.append("Hello"); s.append(" "); s.append("World");
SOPln(s.toString());

### Common Questions
- IsUnique:
	- Use a HashMap, and see if any element has frquency > 1
	- Sort the data structure, and then check for continuous values
- IsPermutation:
	- Use a HashMap, add and then subtract. All should be 0 if it's a permutation.
	- If not, just sort both, and see if they match up char by char. This would be like O(3n)
- URLify [Replace all spaces with %20]
	- To replace in place with a char[], you'll need the correct amount of additional space. Create a new char[] with that space (you may have to iterate once to find it, if not given)
	- Then, go through both arrays, adding '%20' instead of the spaces
	- More efficient than moving the elements each towards the end. <b>That</b> approach would be O(n^2)
- Permutation of a Palindrome:
	 - Use a hashmap to count the number of occurances of each character
	 - All should be even <b>except at most one</b> 
- Compare two strings to check if they're one edit away:
	- Again, either use a HashMap (<b>Do you see a trend here</b>) or just sort, and then check for difference
- String compression["aaaabbbcccc" -> "a4b3c4"]:
	- Just use two pointers, one to the anchor, and one to the end of a set
	- Compress them when pointer != anchor, and the difference+1 is the frequency
- Rotate Matrix:
	- Essentially each column becomes a row. C1 = R'1, C2 = R'2, etc.
	- To do this in place, write a swap function for the above assignments and call it repeatedly

## Linked Lists

- LinkedLists are a collection of objects/structs in continuous memory locations
- You can also iterate through LinkedLists using pointers. In C/C++, it can be done by manipulating pointers
- The "Runner" method is also quite common for LinkedLists
- Accessing: O(n)
- Adding: O(1)
- LinkedLists can also be accessed recursively:

	```java
	void traverse(Node head) {
		if(head != null) {
			processData(head.data); // Here for forwards
			traverse(head.next);
			processData(head.data) // Here for backwards
		}	
	}
	
	Class Node {
	int data;
	Node next = null;
		public Node(int d);
		void appendToTail(int d){
			Node x = new Node(d); 
			Node n = this;
			while(n.next != null) {
				n = n.next;	
			}
			n.next = x;
		}
	}
	```

### Common Questions

- Remove duplicates:
	- Use a HashMap as a temporary buffer
		- If no buffer is allowed, then use MergeSort and then remove duplicates from a sorted LL 
- K'th to the last:
	- Go to the end, to find the length. (length - k) => number of elements to go through. Then, iterate again and return 
- Deleting a node:
	- Go until you find it, and skip: O(n)
- Partition around a value:
	- Use a runner to get to the value that you need to partition around. Then add one from the head, and one from the runner, alternatively
- Intersection/Loop Detection:
	- If you can use C/C++, just use pointers
	- If not, you can still check the reference by == for objects


## Stacks and Queues

### Stacks

- LIFO: <b>Last In, First Out</b>
	pop() -> Pop out the last item
	push() -> Push an item to the top of the stack
	peek() -> Return the top of the stack
	isEmpty() -> Return true iff the stack is empty.

- Stacks do <b>not</b> have constant-time access to the i'th item
- Stacks <b>do</b> have constant-time add and remove ability

Easiest Implementatin: Array/ArrayList -> hold item with index of last item. Use this to pop/peek/append. Remove elements when required.

Another Implementation: Use a LinkedList, and hold the last element as the top element. Then, as you remove that, just reset the next of the second top element to null, and assign that one to top

### Queues

- FIFO: <b>First In, First out</b>
	add(item) -> Adds item to the end of the list
	remove() -> Remove the first item in the list
	peek() -> Returns the top of the queue, but doesn't remove
	isEmpty() -> Returns true iff the queue is empty

- Queues can be <b>directly</b> implemented using LinkedLists.

### Common Questions

- Single Array to maintain 3 stacks:
	- Maintain a single stack, with 0, (len-1)/2, and len-1 being the indexes of the beginnings of the three stacks. The first and third stack would grow in opposite directions, while the second stack would grow in either direction
	- Another way to do this would be to structure it such that there is a secondary array that holds the parent for each node. All the other elements would be <a href="https://prismoskills.appspot.com/lessons/Programming_Puzzles/Multiple_stacks_in_single_array.jsp">linked to the parent</a>
- Stack Min [to always return the minimum element]:
	- Track the min element during insertion
	- Pop each one one by one, until you find the smallest one
- SetOfStacks: 
	- Array of Stacks (or ArrayList). When stacks[i] overflows, move to stacks[i+1]. Track the stack that's at the top, to pop and push from that.
- Queue using 2 Stacks:
	- Solution:
		- <b>To Insert:</b> If s1 != empty => pushAll(s1, s2); Then, pushAll(s2,s1);
		- To dequeue, just pop from s1
		- Ref <a href="https://www.geeksforgeeks.org/queue-using-stacks/">here</a>
- Sort Stack:
	- Poelements out from the input stack. With the popped element, push all the temp elements into the input stacks that are bigger than the element popped from the input stack. Once this is done, push the input pop into the temp stack. Repeat this for every input element. This way, you're basically popping out elements, and recycling them until you find the right one.
	
	```java
	public static Stack<Integer> sortstack(Stack<Integer> input) { 
    	Stack<Integer> tmpStack = new Stack<Integer>(); 
    	while(!input.isEmpty()) { 
        	int tmp = input.pop(); 
        	while(!tmpStack.isEmpty() && tmpStack.peek() > tmp) { 
        		input.push(tmpStack.pop()); 
       		} 
        tmpStack.push(tmp); 
    	} 
    	return tmpStack; 
    }
    ```

## Graphs and Trees

### Trees

- Trees have a root node, and then each root has two or more child notes
- Each child node has 0 or more child nodes, and so on.
- Trees <b>cannot</b> contain cycles
- <b>Worst and last case complexities vary</b>
- <b>Height and Depth</b>"
	- The depth of a node is the number of edges from the node to the tree's root node. A root node will have a depth of 0.
	- The height of a node is the number of edges on the longest path from the node to a leaf. A leaf node will have a height of 0.
- A regular tree has a firstChild object, which generally contains a single child. It also contains a nextSibling LinkedList, with the other siblings

- A binary tree is one where each child has upto only <b>two</b> children
- A binary search tree is one where:
	- The left subtree has nodes with values less than the root
	- The right subtree has nodes with values greater than the root
	- The right and left subtree must also satisfy the same axioms

	```java	
	BinaryNode() {
		int data;
		BinaryNode left;
		BinaryNode right;
	}
	```

- A <b>complete</b> Binary Tree is one where every level is fully filled, except the last one
- A <b>full</b> Binary Tree is one where every node has either zero or two children
- A <b>perfect</b> binary tree is one that's both full, and complete. All the nodes are at the same level, and this last level has the maximum number of nodes.

- Traversals:
	- Post Order
	- Pre Order
	- In Order
	
	```java
	void inOrder(Node node) {
		if (node != null) {
			inOrder(node.left);
			print(node.value);
			inOrder(node.right);
		}
	}

	void postOrder(Node node) {
		if (node != null) {
			postOrder(node.left);
			postOrder(node.right);
			print(node.value);
		}
	}

	void preOrder(Node node) {
		if (node != null) {
			print(node.value);
			preOrder(node.left);
			preOrder(node.right);
		}
	}
	```

#### Expression Trees

- Expression trees are those where the root nodes are operators, and the leaves are the operands
	- They can be evaluated using in-order traversal
	- Prefix Expression: Pre order
	- Postfix Expression: Post order

- Constructing an expression tree from a postfix expression:
	- As you read operators, make them into their own trees and push them onto the stack
	- When you encounter an operand, pop twice, first pop -> right, second pop -> left, operand -> root. Then, push this newly constructed tree onto the stack again
	- Continue this process until either the stack is empty, and the expression has been traversed.
		- If either the Stack has nothing when it needs to have something, or if it's not empty at the end of the expression, then the expression is invalid

#### Binary Search Trees
- For every node, the values of the nodes on the left sub-tree are smaller than their root, and the values of the nodes on the right sub-tree are larger than their root
- Note: Most of these functions return the <b>new root of the sub-tree</b>
- Time taken is dependant on the height: logn to n
- For a BST, checking if an element exists is O(logn) as it's <b>divide and conquer</b> like binary search is

	```java
	boolean contains(root, int x) {
		if (root == null) {
			return false;
		}
		if(root.value.compareTo(x) > 0) {
			return contains(root.left, x);
		} else if(root.value.compareTo(x) < 0) {
			return contains(root.right, x);
		} else {
			return true;
		}
	}
	```

- findMin() traverse the tree to the furtherest left element
- findMax() traverse the tree to the furtherest right element

	```java
	int findMin(BinaryNode root) {
		if(root == null) {
			return null;
		}
		if(root.left == null) {
			return root.value;
		}
		return findMin(root.left);
	}
	```

- To insert into a BST:

	```java
	BinaryNode insert(int data, BinaryNode root) {
		if (root == null) {
			root.data = data;
		}
		if(root.data.compareTo(data) > 0) {
			insert(root.left, data);
		} else if(root.data.compareTo(data) < 0) {
			insert(root.right, data);
		} else {
			System.out.println("Element already exists");
		}
		return root;
	}
	```

- Deleting from a BST is tricky: if you remove an element, you may/may not have to change all the other links and nodes, in order to ensure that the result conforms to the BST definition

	```java
	private BinaryNode removeNode(int x, BinaryNode root) {
		if(t == null) {
			return t;
		}
		if(root.value.compareTo(x) > 0) {
			return contains(root.left, x);
		} else if(root.value.compareTo(x) < 0) {
			return contains(root.right, x);
		} else {
			if(root.left != null && root.right != null) {
				// Here, you find the smallest value on the right, and move that to the root
				// Once that's done, just remove the value that was moved to the root
				int replace = minMin(root.right).value;
				root.value = replace;
				t.right = removeNode(root.right, replace);

			} else {
				root = (root.left != null) ? root.left : root.right;
			}
			// We're at the node we need to remove

		}
	}
	```

- Average case for a BST is O(logn), but the worst case is when you add a list of ascending/descending values. In this case, the BST becomes a LinkedList, and is fairly useless. In order to combat this, you need a <b>self adjusting (AVL) tree</b>

#### AVL Trees
- Eliminate the issue with Binary Trees, which can turn them into O(N) LinkedLists
- Is self adjusting, uses rotations to ensure that it always fulfils a few conditions, which ensure that it's always O(logn)
- Works based on <b>rotations</b>, which basically involve pulling the position of the nodes in order to convert a BST of larger height into one of a lower height
- <b>Balance Factor:</h> h(left subtree) - h(right subtree) = -1, 0, 1 => if this condition is fulfilled, then the node is balanced.
	- |bf| = |hl -  hr| <= 1
- Four types of rotations:
	- Single rotation to the right (LL Imbalance) [LL Rotation]
	- Single rotation to the right to reallign the middle node, followed by a consequent one to the left (RL Imbalance) [RL Rotation]
	- Single rotation to the left (RR Imbalance) [RR Rotation]
	- Single rotation to the left to reallign the middle node, followed by a consequent one to the right (LR Imbalance) [LR Rotation]
- The first two are when hl > hr, and the second two are when hr > hl
- When you're looking at a larger tree, look only at the 3 nodes which matter. Rotate them, rememebr whose subtree one was on. Find the gap, and then add it to that gap.

##### Implementing an AVL Tree
- Insert is the same as for a BST. Insert wherever it fits based on the compareTo(), and then call balance(root)
- Balance is the function that runs rotations, in order for the tree to keep its structure.

	```java
	private AVLNode banace(AVLNode root) {
		if(root == null) {
			return root;
		}

		if(height(t.left)-height(t.right) > 1) {
			if(height(t.left.left) >= height(t.left.right)) {
				// LL imbalance
				t = rotateWithLeftChild(t);
			} else {
				// RL = So you do RL
				t = doubleOnLeftChild(t);
			}
		} else if(height(t.right)-height(t.left) > 1) {
			if(height(t.right.right) >= height(t.right.left)) {
				// RR imbalance
				t = rotateWithRightChild(t);
			} else {
				// LR = So you do LR
				t = doubleOnRightChild(t);
			}
		}

		return t;
	}
	```

#### Level order traversal
- Level order tree traversal of a binary tree

	```java
	class Solution {
	    public List<List<Integer>> levelOrder(TreeNode root) {
			int height = getTreeHeight(root);
			List<List<Integer>> traversal = new ArrayList<List<Integer>>();
			List<Integer> temp = new ArrayList<Integer>();
			for(int i = 1; i <=height; i++) {
				List<Integer> holdElements = new ArrayList<Integer>();
				temp = doLevelOrder(root, i, holdElements);
				traversal.add(temp);
			}
			return traversal;
	    }
	    private List<Integer> doLevelOrder(TreeNode root, int level, List<Integer> myTrav) {
			if(root == null) {
				// This doesn't do much, just serves as an exit
				return null;
			}
			if(level == 1) {
				// You could either add, or ditch the list and just print here. In that case, you wouldn't return
				myTrav.add(root.val);
			}
			doLevelOrder(root.left, level-1, myTrav);
			doLevelOrder(root.right, level-1, myTrav);
			// At this point, the level order has been completed.
			return myTrav;
	    }
	    private int getTreeHeight(TreeNode root) {
			if(root == null) {
				return 0;
			}
			return Math.max(getTreeHeight(root.left), getTreeHeight(root.right)) + 1;
	    }
	}
	```

### Heaps

- A min/max heap is a <b>complete</b> binary tree, where each node has a value that is smaller (in the case of a min heap) or larger (in the case of a max heap) than it's children.
- <b>Heap Order Property:</b> The root is the <b>minimum (MinHeap) or maximum (MaxHeap) element in a tree</b>
- When you insert into a MinHeap, you insert the element into the bottom. You then percolate the element up until it reaches a point where it satisfies the condition of the MinHeap. The process is the same for a MaxHeap too. This process takes O(logn) time.
- When you extract an element, you remove the root and replace it with the most extreme case (bottom right) - then, you percolate down. As you pick between left and right, you pick the smaller one to maintain min-heap properties.


#### Insert (Percolate Up)
- You insert an element at the position that you percolate to

	```java
	void insert(int x) {
		if(currentSize == array.length - 1) {
			enlargeArray(array.length*2 -1);
		}
		// Percolate up until you reach the correct position - you keep moving the nodes/elements down
		int hole = ++currentSize;
		// array[0] is always a placeholder
		for(array[0] = x; x < array[hole/2]; hole/=2) {
			array[hole] = array[hole/2];
		}
		array[hole] = x;
	}
	```

#### Delete (Percolate Down)
- You delete the root for either case. The root is the biggest element in a MaxHeap, and the smallest element in a MinHeap
- The root is replaced by the smaller of the two children for MinHeap, and the larger of the two children for MaxHeap

	```java
	int deleteMin() {
		if(isEmpty()) {
			throw new UnderflowException();
		}

		int min = findMin()
		// You take the smallest/largest element and dump it into the root.
		// Then, you percolate the same down
		array[1] = array[currentSize--];
		percolateDown(1);
	}

	void percolateDown(int hole) {
		// This is for a minheap
		int child;
		int t = array[hole];
		while(hole <= (currentSize/2)) {
			child = hole*2;
			if(child != currentSize && array[child+1] < array[child]) {
				child++;
			}
			if(array[child] < t) {
				array[hole] = array[child];
			} else {
				break;
			}
		}
		array[hole] = temp;
	}
	```

### Tries (Prefix Trees)

- A tree of LinkedLists, it's a variant of an n-arraytree that stores combinations
- For example, each path down the tree can be used to indicate complete words. Each list can be terminated by a special node.
- Generally, a trie can be used to store the entire english language for quick prefix lookups. Since you go down and have various branches (like a tree) it's far more effective than a HashMap is for completion options. A HashMap would be far more efficient for absolute checks.
- A trie can check if a string is a valid prefix in O(n), where n -> length of the string

### AVL Trees
- Trees that automatically do rotations to become BST's - used to combat the issues faced by using BSTs.

### Graphs

- Tree is a type of graph, but not all graphs are trees. Trees are connected graphs without cycles.
- A graph is simply a collection of nodes, with edges between (not necessarily all of) them. 
- Graph edges can be either directed, or undirected. It can have multiple sub-graphs, or not. If every vertex has a path between it, it's a directed graph.
- An <b>acyclic</b> graph is one without cycles.
- Graphs can be represented by either:
	- Adjacency Lists [Space: O(E+N)]:
		- Every node/vertex stores a list of adjancent vertices <br>
		- If the graph is undirected, then an edge between (a,b) would be both in a, and in b's adjacency list<br>
	- Adjacency Matrices [Space: O(|V|^(2)]:
		- An adjacency matrix is an N x M boolean matrix, where a true value of mat[i][j] indicates the presence of an edge from i->j.
		- In an undirected graph, an adjacency matrix will be symmetric. It's not necessarily so in a directed graph
	- The same graph search algorithms used on adjacency lists can be used on matrices, but they may be slightly less efficient. Adjacency lists make it easy to iterate through the neighbours of a node, while adjacency matrices make this slightly more difficult.
	- Adjacency List if the graph is <b>sparse</b>, Adjacency Matrix if it's <b>dense</b>
- Graphs have two main search algorithms: <b>Depth First Search</b> and <b>Breadth First Search</b>
- BFS of a binary tree is the same as an <b>in-order</b> traversal
- DFS of a binary tree is the same as an <b>pre-order</b> traversal
- The edges in a DFS/BFS spanning tree which were present on the original graph but not on the spanning tree -> cross edges
- <a href="https://www.youtube.com/watch?v=pcKY4hjDrxk">For BFS and DFS</a>
- <a href="https://www.youtube.com/watch?v=zaBhtODEL0w">Also for BFS and DFS</a>

#### Breadth First Search
- Algorithm:
	- Pick an arbitrary vertex
	- Explore all the adjacent vertices, to do with that vertex
	- Move on to the next vertex, and visit all adjacent vertices
	- And so on...
- For BFS, you visit all adjacent vertices, before moving on to the next vertex. For DFS you just keep picking vertices and proceeding. You don't visit <b>every</b> unvisited adjacent vertex.
- <b>BFS is NOT recursive</b>
- Implementation of BFS for a graph requires the use of a Queue
	- Pick an arbitrary element, and add it to Q
	- Pop Q, and then explore the vertex. As you explore, once again, add the visited vertices to Q. Output the values explored.
	- Pop the next vertex to explore from Q, and explore the same. Check to see if vertices are in the Q or not, to see if they're already explored.
	- When Q has no more elements to pop, then BFS is complete.
- The tree that results in BFS -> BFS spanning tree. Construct this based on visits, and then the adjacent nodes visited by the exploration.
- When you select a vertex for exploration, <b>explore all the adjacent verticies before moving on to explore the next vertex</b>
- Selecting the next vertex for exploration has to be picked out of a Queue <b>alone</b>

### Topological Sort
- Graph has to be directed, acyclic.
- Since it's a modified DFS, O(V+E)
- Used to get an ordering of the vertices
- Used to get a list of courses to take, in order to comply with a graph of CS pre-reqs
- You can either use root.visited, or use indegrees to track visited nodes
- Ref <a href="https://www.youtube.com/watch?v=Q9PIxaNGnig">here</a>

	```java
	private void topSort(HashMap<Integer, List<Integer>> graph, HashMap<Integer, Integer> inDegrees) throws  CycleFoundException {
		// You could use just a visited marker, but that won't pick up the cycle. This will
		int ctr = 0;        
		Queue<Integer> q = new LinkedList<Integer>();
		for(int x : graph.keySet()) {
			if(inDegrees.get(x) == null) {
				System.out.println("Starting with: "+x);
				q.add(x);
			}
		}
		
		while(!q.isEmpty()) {
			int source = q.poll();
			// You can replace ctr with an Arraylist of the values, if you want to print them
			ctr++;
			for(int x : graph.get(source)) {
				inDegrees.put(x, inDegrees.get(x)-1);
				if(inDegrees.get(x) == 0) {
					q.add(x);
				}
			}
		}
		if(ctr != graph.keySet().size()) {
			throw new CycleFoundException();
		}
	}
	```

### Shortest Path Algorithms
- For an unweighted graph, it's the smallest amount of edges that need to be traversed to go from vertex i to vertex j
- For a weighted graph, it's the minimal costing path that can be taken to go from vertex i to vertex j
- O(V+E)

#### Unweighted Graph:
- A simple <b>breath first search works</b>, where you track the distances

	```java
	void bfs_path(Vertex root) {
		Queue<Vertex> q = new Queue<Vertex>();
		for Vertex x in allVertices {
			x.dist = INFINITY;
		}

		root.dist = 0;
		
		q.enqueue(root);

		while(!q.isEmpty()) {
			Vertex v = q.dequeue();
			List<Vertex> adjList = q.adjList;

			for(Vertex w in adjList) {
				// We use the dist to check if it's been visited
				if(w.dist == INFINITY) {
					w.dist = v.dist + 1;
					w.path = v;
					q.enqueue(w);
				}
			}

		}
	}
	```

	- <b>Note: For both BFS and Topsort, the format is the same. You use a queue, keep checking if it's empty after enqueue-ing the root. You iterate through the adjacent values, and add them to the queue if they fulfil a condition. You also need to mark them as visited somehow</b>

#### Weighted Graph (No negative values) - Dijkstra's Algorithm
- Shortest path between the starting node, and any other node in the graph
- Greedy Algorithm
- O(n^2)
- If the graph is un-weighted, just take all the weights as 1 and you can use BFS. You don't need Dijkstra's
- <b>Dijkstra's doesn't work for negative weights. It may/may not, but you should use Bellman Ford</b>
- The algorithm is fairly simple. As long as an unvisited node exists, find the unvisited node with the smallest distance from s. Then, go through it's neighbours, and run relaxation on each one. Mark a node as vsited as you use it. Within relaxation, set the path of the relaxed node to the previous one in order to get a path later if needed.
- Ref <a href="https://www.youtube.com/watch?v=XB4MIexjvY0">here</a>

- Priority Queue solution (for sparse graphs)

	```java
	void dijkstra(Vertex s) {
    
		for (Vertex v in allVertices) {
			v.dist = INFINITY;
			v.known = false;
		}

		/* 
		Using a priority queue with the distance as the comparison
		allows for us to access the element with the shortest
		distance from the src without an extra loop
		*/
		pq.add(new Node(s.value,0));

		while(!pq.isEmpty()) {

			Vertex v = pq.remove();
			v.known = true;

			// Now, evaluate the neighbours. The PQ must be updated as the distance changes
			List<Vertex> adjVertices = v.adj;

			for(Vertex w in adjVertices) {
				// Visit only if it's not known
				if(!w.known && w.distance != INFINITY) {
					int distance = getCost(v,w);
					// Relaxation step, and set path to w as v
					if(w.dist > v.dist + distance) {
						w.dist = v.dist + distance;
						w.path = v;
					}
					priorityQueue.add(new Node(w,v.dist+distance));
				}
			}
		}
	}
	```

- Regular Solution

	```java
	// unknownExists goes through the nodes, and checks if any have v.known == false
	// getSmallestDistanceUnknownNeighbour goes through the graph's adjacency list, and gets the unknown node with the smallest distance

	void dijkstra(Vertex s) {
	
		for (Vertex v in allVertices) {
			v.dist = INFINITY;
			v.known = false;
		}

		s.dist = 0;

		while(unknownExists(s)) {
			// Get the one with the smallest distance - Greedy
			Vertex v = getSmallestDistanceUnknown(s);
			v.known = true;

			List<Vertex> adjVertices = v.adj;

			for(Vertex w in adjVertices) {
				// Visit only if it's not known
				if(!w.known) {
					int distance = getCost(v,w);
					// Relaxation step, and set path to w as v
					if(w.dist > v.dist + distance) {
						w.dist = v.dist + distance;
						w.path = v;
					}
				}
			}
		}
	}
	```

#### Weighted Graph (Negative values) - Bellman Ford
- Meant to fix Dijkstra's biggest drawback: negative edges
- Uses Dynamic Programming, relax all edges |V| - 1 times
- You cover all possible paths
Algorithm:
- Prepare a list of all possible edges
- Mark the distance of all the possible edges from the source as infinity
- Then, start relaxing the edges. Repeat this process |V|-1 times
- O(VE) 
	- O(n^2) -> If it's nxn
	- O(n^3) -> For a complete graph
- Bellman Ford fails for graphs with negative weight cycles
	- How do you detect? After |V|-1 relaxations, relax one more time. If there's another change, then there's a negative edge cycle

	```java
	void dijkstra(Vertex s) throws NegativeCycleException {
		for (Vertex v in allVertices) {
			v.dist = INFINITY;
			v.known = false;
		}

		s.dist = 0;

		while(unknownExists(s)) {
			// Get the one with the smallest distance - Greedy
			Vertex v = getSmallestDistanceUnknown(s);
			v.known = true;

			// Run it v-1 times
			for(int i = 1; i < VERT_COUNT; i++) {
				for(Edge e : getEdges()) {
					Node u = e.source;
					Node v = e.target;
					// Check, and set prev as well
					relax(u,v);
				}
			}

			// This is to check for the presence of negative edge cycles
			for(Edge e : getEdges()) {
				Node u = e.source;
				Node v = e.target;
				if(checkForRelax(u,v)) {
					throw new NegativeCycleException();
				}
			}
			// At this point, we should have it implemented. 
		}
	}
	```


#### Depth First Search
- Allows you to visit nodes and edges of a graph
- O(V+E)
- Isn't that useful by itself, but while augmented, it can perform things like counting connected components, determining connectivity, etc
- You literally plunge into any node, and traverse all the edges, backtracking when you can't go to any other node from a position. Once you backtrack, <b>then</b> you continue the exploration.

	```python
	n = (nodes)
	g = [adjacency list]
	v = [false, false, false,...] # Visited

	function dfs(at):
		ivf visited[at]:
			# If already visited, then just return
			return
		# Do whatever you want to do here
		visited[at] = True
		# Get the neighbours, and go thorugh each one
		neighbours = g[at]
		for node in neighbours:
			# DFS each neighbour node
			dfs(node)

	start = 0
	dfs(0)
	```

#### Comparing DFS vs BFS

| DFS | BFS |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| In DFS, we might traverse through more edges to reach a destination vertex from a source. | BFS can be used to find the single source shortest path in an unweighted graph.  We reach a vertex with minimum number of edges from a source vertex. |
| DFS is more suitable when there are solutions away from source. | BFS is more suitable for searching verteces which are closer to the given source. |
| DFS is more suitable for game or puzzle problems. We make a decision, then explore all paths through this decision. And if this decision leads to win situation, we stop. | BFS considers all neighbors first and therefore not suitable for decision making trees used in games or puzzles. |

- Finding connected components:
	- Counting multiple disjoint componenets. Start dfs at [0,n] and find the number of connected elements. This helps you find multiple connected componenets
	
	```python
	n = (nodes)
	g = [adjacency list]
	v = [false, false, false,...] # Visited
	count = 0
	function dfs(at):
		if visited[at]:
			# If already visited, then just return
			return
		visited[at] = True
		components[at] = count
		# Get the neighbours, and go thorugh each one
		neighbours = g[at]
		for node in neighbours:
			# DFS each neighbour node
			dfs(node)
	def findComponents():
		for(i in range(0, n)):
			if !visited[i]:
				count += 1
				dfs(i)
		return (count, componenets)
	```

- Another implementation of DFS involves using a Stack to track vertices which we're yet to explore.
	- Visit an arbitraty vertex, and find any <b>one</b> connected vertex.
	- Push the original vertex to the Stack, and start exploring the vertex you just navigated to
	- Carry this on, suspending vertices and adding them to the Stack. When you don't have any edges on a vertex, <b>revert</b> to an earlier vertex by popping the stack

##### Problems uses

- Deepclone a graph (using BFS):

	```java
	public Node cloneGraph(Node node) {
        HashMap<Node,Node> visited = new HashMap<Node,Node>();
        Queue<Node> q = new LinkedList<Node>();
        
        visited.put(node, new Node(node.val, new ArrayList()));
        q.add(node);
        
        while(!q.isEmpty()) {
            Node src = q.poll();
            Node target = visited.get(src);
            
            if(src.neighbors != null) {
                List<Node> nn = src.neighbors;
                for(Node x : nn) {
                    Node subTarget = visited.get(x);
                    if(subTarget == null) {
                        subTarget = new Node(x.val, new ArrayList());
                        visited.put(x, subTarget);
                        q.add(x);
                    }
                    target.neighbors.add(subTarget);
                }
            }
        }
        return visited.get(node);
    }
	```

- Using DFS to find cycles (Can also use TopSort):

	```python
	def isCyclicUtil(v, visited, recStack): 
		visited[v] = True
        recStack[v] = True
        # Recur for all neighbours - if any neighbour is visited and in recStack then graph is cyclic 
        for neighbour in graph[v]: 
            if visited[neighbour] == False: 
                if isCyclicUtil(neighbour, visited, recStack) == True: 
                    return True
            elif recStack[neighbour] == True: 
                return True
        # The node needs to be poped from the recursion stack before function ends 
        recStack[v] = False
        return False
    # Returns true if graph is cyclic
    # You essentially BFS, and see if the same node is visited multiple times - if so, this implies that the graph is cyclic
    def isCyclic(): 
        visited = [False] * V 
        recStack = [False] * V 
        for node in range(V): 
            if visited[node] == False: 
                if isCyclicUtil(node,visited,recStack) == True: 
                    return True
        return False
    ```

- Some other uses: Minimum spanning tree, check for bipartite, check for strongly connected components, topological sorting, generate mazes.

### Prim's Algorithm
- <a href="https://www.youtube.com/watch?v=4ZlRH0eK-qQ">Ref for Prim's and Kruskal's</a>
- Used to find the minimum spanning tree from a graph
- Algorithm:
	- Always the minimum edge that's connected to the starting elementB

### Kruskal's Algorithm
- Always select the minimum cost edge, they don't have to be connected to each other
- If an edge is forming a cycle, then <b>don't</b> select it
- O(n2) without MinHeap, O(nlogn) with MinHeap

### Bidirectional Search
- Part of Dijkstra's Algorithm
- Used to find the shortest path between a source and destination node
	- You esseentially run two BFS-es, one from each node.
	- When their searches collide, we've found a path
- Reduces the amount of exploration needed. Ref <a href="https://www.geeksforgeeks.org/bidirectional-search/">here</a>


## Binary Search
- O(logn) complexity
- Divide and conquer
- Splits the array based on the middle value, and keeps searching
- Works for sorted arrays

	```python
	def binarySearch(arr, x):
		l = 0
		r = len(arr)-1
		# Loop while left is less than right
		while(l < r):
			# Set the middle, work based on this
			mid = l+((r-l)/2)
			# If you find it, get out
			if arr[mid] == x:
				return mid
			# If the middle is too big, use the first half
			else if arr[mid] > x:
				r = mid-1
			# If the middle is too small, use the second half
			else:
				l = mid+1
		# If nothing worked, it doesn't exist
		return -1

	def rec_binarySearch(arr, left, right, x): 
		# Check base case
		if(l > r):
			return -1
		mp = (r - l)/2
		mid = l + mp
		# If element is present at the middle itself 
		if arr[mid] == x: 
			return mid 
		# If element is smaller than mid, then it  
		# can only be present in left subarray 
		elif arr[mid] > x: 
			return binarySearch(arr, l, mid-1, x) 
		# Else the element can only be present  
		# in right subarray 
		else: 
			return binarySearch(arr, mid+1, r, x) 
	```

# Topic 2: Sorting Algorithms

A list of useful sorting algorithms

## Selection Sort

- Selection sort finds the minimum element in the unsorted part of the array and swaps it with the first element in the unsorted part of the array.
- The sorted part of the array grows from left to right with every iteration.
- After `i` iterations, the first `i` elements of the array are sorted.
- Sorts in-place. Not stable.<sup>[1](#footnote1)</sup>

### Algorithm

```java
void selectionSort(int[] arr) {
    for (int i = 0; i < arr.length; i++) {
        int min = i;
        for (int j = i; j < arr.length; j++) {
            if (arr[j] <= arr[min]) min = j;
        }

        if (min != i) {
            swap(arr, i, min);
        }
    }
}
```

### Time Complexity

- **Best Case:** `O(n^2)`
- **Average Case:** `O(n^2)`
- **Worst Case:** `O(n^2)`

## Bubble Sort

- In every iteration, bubble sort compares every couplet, moving the larger element to the right as it iterates through the array.
- The sorted part of the array grows from right to left with every iteration.
- After `i` iterations, the last `i` elements of the array are the largest and sorted.
- Sorts in-place. Stable.<sup>[1](#footnote1)</sup>

### Algorithm

```java
void bubbleSort(int[] arr) {
    boolean swapped = true;
    int j = 0;

    while (swapped) {
        swapped = false;
        for (int i = 1; i < arr.length - j; i++) {
            if (arr[i - 1] > arr[i]) {
                swap(arr, i - 1, i);
                swapped = true;
            }
        }
        j++;
    }
}
```

### Time Complexity

- **Best Case:** `O(n)`
- **Average Case:** `O(n^2)`
- **Worst Case:** `O(n^2)`

## Insertion Sort

- In every iteration, insertion sort takes the first element in the unsorted part of the array, finds the location it belongs to within the sorted part of the array and inserts it there.
- The sorted part of the array grows from left to right with every iteration.
- After `i` iterations, the first `i` elements of the array are sorted.
- Sorts in-place. Stable.<sup>[1](#footnote1)</sup>

### Algorithm

```java
void insertionSort(int[] arr) {
    for (int i = 1; i < arr.length; i++) {
        for (int j = i; j > 0; j--) {
            if (arr[j - 1] > arr[j]) {
                swap(arr, j - 1, j);
            } else {
                break;
            }
        }
    }
}
```

### Time Complexity

- **Best Case:** `O(n)`
- **Average Case:** `O(n^2)`
- **Worst Case:** `O(n^2)`

## Merge Sort

- Uses the *divide & conquer* approach.
- Merge sort divides the original array into smaller arrays recursively until the resulting subarrays have one element each.
- Then, it starts merging the divided subarrays by comparing each element and moving the smaller one to the left of the merged array.
- This is done recursively till all the subarrays are merged into one sorted array.
- Requires `O(n)` space. Stable.<sup>[1](#footnote1)</sup>

### Algorithm

```java
void mergesort(int[] arr) {
    int[] helper = new int[arr.length];
    mergesort(arr, helper, 0, arr.length - 1);
}

void mergesort(int[] arr, int[] helper, int low, int high) {
    // Check if low is smaller than high, if not then the array is sorted.
    if (low < high) {
        int mid = low + ((high - low) / 2);        // Get index of middle element
        mergesort(arr, helper, low, mid);          // Sort left side of the array
        mergesort(arr, helper, mid + 1, high);     // Sort right side of the array
        merge(arr, helper, low, mid, high);        // Combine both sides
    }
}

void merge(int[] arr, int[] helper, int low, int mid, int high) {
    // Copy both halves into a helper array.
    for (int i = low; i <= high; i++) {
        helper[i] = arr[i];
    }

    int helperLeft = low;
    int helperRight = mid + 1;
    int current = low;

    // Iterate through helper array. Compare the left and right half, copying back
    // the smaller element from the two halves into the original array.
    while (helperLeft <= mid && helperRight <= high) {
        if (helper[helperLeft] <= helper[helperRight]) {
            arr[current] = helper[helperLeft];
            helperLeft++;
        } else {
            arr[current] = helper[helperRight];
            helperRight++;
        }
        current++;
    }

    // Copy the rest of the left half of the array into the target array. Right half
    // is already there.
    while (helperLeft <= mid) {
        arr[current] = helper[helperLeft];
        current++;
        helperLeft++;
    }
}
```

### Time Complexity

- **Best Case:** `O(n log n)`
- **Average Case:** `O(n log n)`
- **Worst Case:** `O(n log n)`

## Quicksort

- Quicksort starts by selecting one element as the *pivot*. The array is then divided into two subarrays with all the elements smaller than the pivot on the left side of the pivot and all the elements greater than the pivot on the right side.
- It recursively repeats this process on the left side until it is comparing only two elements at which point the left side is sorted.
- Once the left side is sorted, it performs the same recursive operation on the right side.
- Quicksort is the fastest general purpose in-memory sorting algorithm in practice.
    - Best case occurs when the pivot always splits the array into equal halves.
    - Usually used in conjunction with Insertion Sort when the subarrays become smaller and *almost* sorted.
- Requires `O(log n)` space on average. Not stable.<sup>[1](#footnote1)</sup>

### Algorithm

```java
void startQuicksort(int[] arr) {
    quicksort(arr, 0, arr.length - 1);
}

void quicksort(int[] arr, int low, int high) {
    if (low >= high) return;

    int mid = low + ((high - low) / 2);
    int pivot = arr[mid];  // pick pivot point

    int i = low, j = high;
    while (i <= j) {
        // Find element on left that should be on right.
        while (arr[i] < pivot) i++;

        // Find element on right that should be on left.
        while (arr[j] > pivot) j--;

        // Swap elements and move left and right indices.
        if (i <= j) {
            swap(arr, i, j);
            i++;
            j--;
        }
    }

    // Sort left half.
    if (low < i - 1)
        quicksort(arr, low, i - 1);

    // Sort right half.
    if (i < high)
        quicksort(arr, i, high);
}
```

### Time Complexity

- **Best Case:** `O(n log n)`
- **Average Case:** `O(n log n)`
- **Worst Case:** `O(n^2)`

## Bucket Sort

- Bucket sort is a sorting algorithm that works by distributing the elements of an array into a number of buckets.
- Each bucket is then sorted individually, either using a different sorting algorithm or by recursively applying the bucket sorting algorithm.

### Time Complexity

- **Average Case:** `O(n + k)` (where `k` is the number of buckets)
- **Worst Case:** `O(n^2)`

## Counting Sort
- Using a HashMap to sort an array based on a pre-defined order

- **Average Case:** `O(n)` 
- **Worst Case:** `O(n)`

## Heap Sort

- Heap sort takes the maximum element in the array and places it at the end of the array.
- At every iteration, the maximum element from the unsorted part of the array is selected by taking advantage of the binary heap data structure and placed at the end. Then, the unsorted part is heapified and the process is repeated.
- After `i` iterations, the last `i` elements of the array are sorted.
- Sorts in-place. Not stable.<sup>[1](#footnote1)</sup>

### Binary Heap (Array Implementation)

- We can implement a binary heap with `n` nodes using an array with the following conditions:
    - The left child of `nodes[i]` is `nodes[2i + 1]`.
    - The right child of `nodes[i]` is `nodes[2i + 2]`.
    - `nodes[i]` is a leaf if `2i + 1` > `n`.
- Therefore, in a binary max heap, `nodes[i]` > `nodes[2i + 1]` & `nodes[i]` > `nodes[2i + 2]`.

### Algorithm

```java
void heapSort(int[] arr) {
    int n = arr.length;

    // Construct initial max-heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // Extract an element one by one from heap
    for (int i = n - 1; i >= 0; i--) {
        // Move current root to end
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        // Call heapify() on the reduced heap
        heapify(arr, i, 0);
    }
}

// Heapifies a subtree rooted at arr[i]. n is the size of the entire heap.
void heapify(int arr[], int n, int i) {
    int largest = i;  // initialize largest as root
    int l = 2*i + 1;  // left child
    int r = 2*i + 2;  // right child

    // If left child is larger than root
    if (l < n && arr[l] > arr[largest])
        largest = l;

    // If right child is larger than largest so far
    if (r < n && arr[r] > arr[largest])
        largest = r;

    // If largest is not root
    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;

        // Recursively heapify the affected subtree
        heapify(arr, n, largest);
    }
}
```

### Time Complexity

- **Best Case:** `O(n log n)`
- **Average Case:** `O(n log n)`
- **Worst Case:** `O(n log n)`


## Radix Sort

- Radix sort is a sorting algorithm for integers (and some other data types) that groups the numbers by each digit from left to right (most significant digit radix sort) or right to left (least significant digit radix sort) on every pass.
- This process is repeated for each subsequent digit until the whole array is sorted.

### Time Complexity

- **Worst Case:** `O(kn)` (where `k` is the number of passes of the algorithm)

---

<a name="footnote1">[1](#footnote1)</a>: A sorting algorithm is said to be **stable** if two objects with equal keys appear in the same order in sorted output as they appear in the input array to be sorted.

















