package P3;

import static org.junit.Assert.*;

import org.junit.Test;

public class FriendshipGraphTest {

	
	@Test(expected=AssertionError.class)
    public void testAssertionsEnabled() {
        assert false;
       
    }
	/**
	 * if the related boolean value in the adjacent matrix is consistent
	 */
	@Test
	public void testaddEdge() {
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
		boolean[][] expected=new boolean[1000][1000];
		expected[0][1]=true;
		expected[1][0]=true;
		expected[1][2]=true;
		expected[2][1]=true;
		assertArrayEquals(expected, graph.get_matrix());
	}
	/**
	 * if the value of size is right which shows the number of people
	 */
	@Test
	public void testaddVertex() {
		FriendshipGraph graph = new FriendshipGraph();
		Person rachel = new Person("Rachel");
		Person ross = new Person("Ross");
		Person ben = new Person("Ben");
		Person kramer = new Person("Kramer");
		graph.addVertex(rachel);
		assertEquals(1, graph.get_size());
		graph.addVertex(ross);
		assertEquals(2, graph.get_size());
		graph.addVertex(ben);
		assertEquals(3, graph.get_size());
		graph.addVertex(kramer);
		assertEquals(4, graph.get_size());
	}
	

	
}
