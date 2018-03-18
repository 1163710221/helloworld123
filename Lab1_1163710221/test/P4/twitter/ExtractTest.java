/**
 * 
 */
package P4.twitter;
import static org.junit.Assert.*;

import java.awt.List;
import java.lang.reflect.Array;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.junit.Test;

import P4.twitter.Extract;
import P4.twitter.Timespan;
import P4.twitter.Tweet;

/**
 * @author Dell
 *
 */
public class ExtractTest {

	/**
	 * test time span is correct
	 * Tweet(final long id,
	 *  final String author, 
	 *  final String text, 
	 *  final Instant timestamp
	 */
	@Test
	public void testgetTimespan() {
		Instant one=Instant.now();
		Instant two=Instant.now().plusSeconds(3600);
		Instant three=Instant.now().plusSeconds(7200);
		Tweet test1=new Tweet(1,"harry","nothing",one);
		Tweet test2=new Tweet(2,"harry","nothing",one);
		Tweet test3=new Tweet(3,"harry","nothing",one);
		Tweet test4=new Tweet(4,"harry","nothing",two);
		Tweet test5=new Tweet(5,"harry","nothing",three);
		ArrayList<Tweet> tweet_list = new ArrayList<>();
		tweet_list.add(test1);
		tweet_list.add(test2);
		tweet_list.add(test3);
		assertEquals("Correct", new Timespan(one, one), Extract.getTimespan(tweet_list));
		tweet_list.add(test4);
		tweet_list.add(test5);
		assertEquals("Correct", new Timespan(one, three), Extract.getTimespan(tweet_list));
		
		
	}
	/**
	 * test getname is correct or not
	 */
	@Test
	public void testgetName() {
		String String1="\n\n@HarRy";
		String String2="\t@HarrY";
		String String3="@HArry:";
		String String4="Hello@harry";
		String String5="Hello@HARry.com";
		String String6="@Harry£ºjjasd";
		String String7="@Harry\n";
		assertEquals("HarRy", Extract.getName(String1));
		assertEquals("HarrY", Extract.getName(String2));
		assertEquals("HArry", Extract.getName(String3));
		assertEquals("NULL", Extract.getName(String4));
		assertEquals("NULL", Extract.getName(String5));
		assertEquals("Harry", Extract.getName(String6));
		assertEquals("Harry", Extract.getName(String7));
	}
	/**
	 * test  Mentioned names
	 */
	@Test
	public void testgetMentionedUsers() {
		String[] string= new String[8];
		 string[1]="\n\n@APple";
		 string[2]="\t@BeCuals";
		string[3]="@Apple";
		 string[4]="Hello@Kate";
		 string[5]="Hello@Kate.com";
		 string[6]="@KAte£ºjjasd";
		 string[7]="@Kate\n";
		Set<String> Users=new HashSet<String>();
		Users.add("APple");
		Users.add("BeCuals");
		Users.add("KAte");
		
	
		ArrayList<Tweet> lis=new ArrayList<Tweet>();
		for(int i=1;i<=7;i++) {
			lis.add(new Tweet(1, "rao",string[i],Instant.now()));
		}
		assertEquals(Users, Extract.getMentionedUsers(lis));

	}

}
