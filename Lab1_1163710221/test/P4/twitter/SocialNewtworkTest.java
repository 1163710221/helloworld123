package P4.twitter;

import static org.junit.Assert.*;

import org.junit.Test;

import P4.twitter.SocialNetwork;
import P4.twitter.Tweet;

import java.security.GuardedObject;
import java.security.acl.Owner;
import java.time.Instant;
import java.util.*;
public class SocialNewtworkTest {
	/**
	 * partition: case;
	 */
	@Test
	public void testSocialNetwork() {
		List<Tweet> tweets = new ArrayList<Tweet>();
		tweets.add(new Tweet(1, "Harry", "Hello young @lady", Instant.now()));
		tweets.add(new Tweet(2, "Harry", "HEllo @young lady", Instant.now()));
		tweets.add(new Tweet(3, "Harry", "@hellO young lady", Instant.now()));
		HashSet<String> set = new HashSet<String>();
		set.add("lady");
		set.add("young");
		set.add("hellO");
		Map<String, HashSet<String>> hash= new HashMap<String,HashSet<String>>();
		hash.put("Harry", set);
		assertEquals(hash, SocialNetwork.guessFollowsGraph(tweets));
	}
	/**
	 * test the ranking
	 */
	@Test
	public void testfluencers() {
		List<Tweet> tweets = new ArrayList<Tweet>();
		tweets.add(new Tweet(1, "Harry", "Hello young @lady", Instant.now()));
		tweets.add(new Tweet(2, "Harry", "HEllo @young lady", Instant.now()));
		tweets.add(new Tweet(4, "young", "hellO young @lady", Instant.now()));
		tweets.add(new Tweet(5, "young", "@Harry young @lady", Instant.now()));
		tweets.add(new Tweet(6, "lady", "hellO @young", Instant.now()));
		LinkedList<String> set_string = new LinkedList<String>();
		set_string.add("lady");
		set_string.add("young");
		set_string.add("Harry");
		assertEquals(set_string, SocialNetwork.influencers(SocialNetwork.guessFollowsGraph(tweets)));
		
		
		
	}
	/**
	 * test the closure
	 */
	@Test
	public void testTria() {
		List<Tweet> tweets = new ArrayList<Tweet>();
		tweets.add(new Tweet(1, "A", "@B", Instant.now()));
		tweets.add(new Tweet(2, "B", "@C", Instant.now()));
		tweets.add(new Tweet(3, "C", "hello", Instant.now()));
		Set<String> expected = new HashSet<String>();
		expected.add("C");
		expected.add("B");
		Map<String, Set<String>> hash= SocialNetwork.guessFollowsGraph(tweets);
		SocialNetwork.triadic(hash);
		assertEquals(expected, hash.get("A"));
		
	}

}
