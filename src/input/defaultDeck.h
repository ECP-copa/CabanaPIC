#ifndef DEFAULT_INPUT_DECK_H
#define DEFAULT_INPUT_DECK_H
#include "input/deck.h"

class defaultDeck
{
public:
    using real_ = real_t;
    Input_Deck* deck;

    defaultDeck()
    {
	deck = new Input_Deck;
    }
    
    ~defaultDeck()
    {
	delete deck;
    }
    
    void Create() {
	std::cout<< "Default Input Deck\n";
        deck->derive_params();
        deck->print_run_details();
	
    }
};


#endif
