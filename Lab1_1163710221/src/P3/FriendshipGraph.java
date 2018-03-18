package P3;

import java.lang.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

public class FriendshipGraph {
	private ArrayList<Person> namepool = new ArrayList<Person>();
	private boolean[][] adj_m = new boolean[1000][1000];

	private int size = 0;

	private int at(Person person) {	
		int counter = -1;
		for (Person temp : namepool) {
			counter++;
			if (temp.Name().equals(person.Name())) {
				return counter;
			}

		}
		return counter;
	}
	/**
	 * 
	 * @return get the adjacent matrix
	 */
	public boolean[][] get_matrix(){
		return adj_m;
	}
	/**
	 * 
	 * @param person which contains the name of person
	 */
	public void addVertex(Person person) {
		try {
			size++;
			for (Person temp : namepool) {
				if (temp.Name().equals(person.Name()))
					throw new Exception("Error!!same name");
			}
			namepool.add(person);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	public int get_size() {
		return this.size;
	}
	/**
	 * 
	 * @param p1 
	 * @param p2
	 * add an edge from p1 to p2
	 */
	public void addEdge(Person p1, Person p2) {
		int index_1, index_2;
		index_1 = at(p1);
		index_2 = at(p2);
		adj_m[index_1][index_2] = true;

	}
	/**
	 * 
	 * @param p1
	 * @param p2
	 * @return the distance from p1 to p2, if they are the same person then returns 0
	 * if there are non-related then return -1
	 */
	public int getDistance(Person p1, Person p2) {
		if (p1.equals(p2))
			return 0;
		int index_1, index_2;
		int dis = 0;
		index_1 = at(p1);
		index_2 = at(p2);
		boolean[] visited = new boolean[size];
		LinkedList<Integer> que = new LinkedList<Integer>();
		que.addLast(index_1);
		visited[index_1] = true;
		int[] path = new int[size];
		for (int i = 0; i < size; i++)
			path[i] = -1;
		for (int k = 0; k < size + 4; k++) {
			if (que.isEmpty())
				break;
			int start = que.removeFirst();
			for (int i = 0; i < size; i++) {
				if (adj_m[start][i] && !visited[i]) {
					que.addLast(i);
					visited[i] = true;
					path[i] = start;

				}
			}
		}
		int end = index_2;
		if (path[end] < 0)
			return -1;
		while (path[end] >= 0) {
			end = path[end];
			dis++;
		}
		return dis;

	}
	public static void main(String[] args) {
		FriendshipGraph graph = new FriendshipGraph();
		Person rachel = new Person("Rachel");
		Person ross = new Person("Ross");
		Person ben = new Person("Ben");
		Person kramer = new Person("Kramer");
		graph.addVertex(rachel);
		graph.addVertex(ross);
		graph.addVertex(ben);
		graph.addVertex(kramer);
		graph.addEdge(rachel, ross);
		graph.addEdge(ross, rachel);
		graph.addEdge(ross, ben);
		graph.addEdge(ben, ross);
		boolean[][] adj=graph.get_matrix();
		System.out.println(graph.getDistance(rachel, ross));
		System.out.println(graph.getDistance(rachel, ben));
		System.out.println(graph.getDistance(rachel, rachel));
		System.out.println(graph.getDistance(rachel, kramer));
	}

}
