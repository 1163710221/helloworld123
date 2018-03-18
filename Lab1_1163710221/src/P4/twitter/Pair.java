package P4.twitter;
/**
 * 
 * @author Yanghan
 * 
 */
public class Pair {
	private String first;
	private int second;
	public Pair(String first,int second) {
		this.first=first;
		this.second=second;
	}
	public String get_first() {
		return first;
	}
	public int get_second() {
		return second;
	}
	public void increment() {
		second++;
	}
	public void newvalue(int value) {
		second=value;
	}

}
