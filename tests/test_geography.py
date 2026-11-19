from apartworld_ai_assistent.geography import NEIGHBOURS, district_key, proximity, requested_districts


def test_district_key_takes_name_after_dot():
    assert district_key("Prague 5 · Smíchov") == "smichov"


def test_requested_districts_from_free_text():
    assert requested_districts(["near Smíchov", "quiet"]) == {"smichov"}
    assert requested_districts(["SMICHOV or Vinohrady"]) == {"smichov", "vinohrady"}
    assert requested_districts(["Nové Město"]) == {"nove mesto"}
    assert requested_districts(["balcony"]) == set()


def test_part_of_a_word_is_not_a_district():
    # "karlin" must not match inside another word
    assert requested_districts(["karlinsky"]) == set()


def test_neighbours_are_symmetric():
    for district, near in NEIGHBOURS.items():
        for other in near:
            assert district in NEIGHBOURS[other], (district, other)


def test_smichov_and_holesovice_are_not_neighbours():
    assert proximity("Prague 7 · Holešovice", {"smichov"}) == "other"
    assert proximity("Prague 2 · Nové Město", {"smichov"}) == "nearby"
    assert proximity("Prague 5 · Smíchov", {"smichov"}) == "requested"
