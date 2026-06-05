from Models import Source
from pydantic import BaseModel


class Performer(BaseModel):

    def check_overlap(self, source_a: Source, source_b: Source) -> float:
        """ Get the communs index characters between source_a and source_b,
            and return the percent of communs characters """
        if source_a.file_path != source_b.file_path:
            return 0.0

        # Create sets with all the indexs between start and end for each source
        set_a = set(range(source_a.first_character_index,
                          source_a.last_character_index))
        set_b = set(range(source_b.first_character_index,
                          source_b.last_character_index))

        # Get communs indexs with intersection (&)
        nb_communs_characters = len(set_a & set_b)

        if nb_communs_characters == 0:
            return 0.0

        percent_a = nb_communs_characters / len(set_a)
        percent_b = nb_communs_characters / len(set_b)

        return max(percent_a, percent_b)

# a continuer apres le search dataset
