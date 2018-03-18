package P4.twitter;

import static org.junit.Assert.*;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import javax.annotation.Generated;

import org.junit.Test;

import P4.twitter.Filter;
import P4.twitter.Timespan;
import P4.twitter.Tweet;

public class FilterTest {
	
	public List<Tweet> buildtest1() {
		List<Tweet> tweets= new ArrayList<Tweet>();
		tweets.add(new Tweet(1, "Harry", " ", Instant.now()));
		tweets.add(new Tweet(2, "Harry", " ", Instant.now()));
		tweets.add(new Tweet(3, "Kate", " ", Instant.now()));
		tweets.add(new Tweet(4, "Carry", " ", Instant.now()));
		return tweets;
		
	}
	/**
	 * partition:case
	 */
	@Test
	public void testWrittenby() {
		List<Tweet> tweets = new ArrayList<Tweet>();
		List<Tweet> another = new ArrayList<Tweet>();
		tweets.add(new Tweet(1, "Harry", " ", Instant.now()));
		tweets.add(new Tweet(2, "HaRry", " ", Instant.now()));
		another.add(new Tweet(1, "Harry", " ", Instant.now()));
		another.add(new Tweet(2, "Harry", " ", Instant.now()));
		tweets.add(new Tweet(3, "Kate", " ", Instant.now()));
		tweets.add(new Tweet(4, "Carry", " ", Instant.now()));
		assertEquals(another,Filter.writtenBy(tweets, "HARRy"));
		
		
	}
	/**
	 * partition same timestamp?
	 */
	@Test
	public void testIntimespan() {
		List<Tweet> tweets = new ArrayList<Tweet>();
		List<Tweet> another = new ArrayList<Tweet>();
		Instant one = Instant.now();
		Instant two = Instant.now().plusSeconds(30);
		Instant three = Instant.now().plusSeconds(60);
		Instant four = Instant.now().plusSeconds(90);
		another.add(new Tweet(2, "Harry", " ",two));
		another.add(new Tweet(3, "Harry", " ", three));
		tweets.add(new Tweet(1, "Harry", " ", one));
		tweets.add(new Tweet(2, "Harry", " ", two));
		tweets.add(new Tweet(3, "Harry", " ", three));
		tweets.add(new Tweet(4, "Carry", " ", four));
		assertEquals(another,Filter.inTimespan(tweets,new Timespan(one,four)));
		
		
	}
	/**
	 * partition:case
	 */
	@Test
	public void testword() {
		List<Tweet> tweets = new ArrayList<Tweet>();
		List<String> words = new ArrayList<String>();
		words.add("hello");
		tweets.add(new Tweet(1, "Harry", "Hello young lady", Instant.now()));
		tweets.add(new Tweet(2, "Harry", "HEllo young lady", Instant.now()));
		tweets.add(new Tweet(3, "Harry", "hellO young lady", Instant.now()));
		assertEquals(tweets, Filter.containing(tweets,words ));
		
	}

}
