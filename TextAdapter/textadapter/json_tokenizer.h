/* Adapted from json.org JSON_checker */

#ifndef JSON_TOKENIZER_H
#define JSON_TOKENIZER_H

#define __   -1     /* the universal error code */

enum classes {
    C_SPACE,  /* space */
    C_NEWLINE, /* newline */
    C_WHITE,  /* other whitespace */
    C_LCURB,  /* {  */
    C_RCURB,  /* } */
    C_LSQRB,  /* [ */
    C_RSQRB,  /* ] */
    C_COLON,  /* : */
    C_COMMA,  /* , */
    C_QUOTE,  /* " */
    C_BACKS,  /* \ */
    C_SLASH,  /* / */
    C_PLUS,   /* + */
    C_MINUS,  /* - */
    C_POINT,  /* . */
    C_ZERO ,  /* 0 */
    C_DIGIT,  /* 123456789 */
    C_LOW_A,  /* a */
    C_LOW_B,  /* b */
    C_LOW_C,  /* c */
    C_LOW_D,  /* d */
    C_LOW_E,  /* e */
    C_LOW_F,  /* f */
    C_LOW_L,  /* l */
    C_LOW_N,  /* n */
    C_LOW_R,  /* r */
    C_LOW_S,  /* s */
    C_LOW_T,  /* t */
    C_LOW_U,  /* u */
    C_ABCDF,  /* ABCDF */
    C_E,      /* E */
    C_ETC,    /* everything else */
    NR_CLASSES
};

static int ascii_class[128] = {
/*
    This array maps the 128 ASCII characters into character classes.
    The remaining Unicode characters should be mapped to C_ETC.
    Non-whitespace control characters are errors.
*/
    __,      __,      __,      __,      __,      __,      __,      __,
    __,      C_WHITE, C_NEWLINE, __,      __,      C_WHITE, __,      __,
    __,      __,      __,      __,      __,      __,      __,      __,
    __,      __,      __,      __,      __,      __,      __,      __,

    C_SPACE, C_ETC,   C_QUOTE, C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,
    C_ETC,   C_ETC,   C_ETC,   C_PLUS,  C_COMMA, C_MINUS, C_POINT, C_SLASH,
    C_ZERO,  C_DIGIT, C_DIGIT, C_DIGIT, C_DIGIT, C_DIGIT, C_DIGIT, C_DIGIT,
    C_DIGIT, C_DIGIT, C_COLON, C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,

    C_ETC,   C_ABCDF, C_ABCDF, C_ABCDF, C_ABCDF, C_E,     C_ABCDF, C_ETC,
    C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,
    C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_ETC,
    C_ETC,   C_ETC,   C_ETC,   C_LSQRB, C_BACKS, C_RSQRB, C_ETC,   C_ETC,

    C_ETC,   C_LOW_A, C_LOW_B, C_LOW_C, C_LOW_D, C_LOW_E, C_LOW_F, C_ETC,
    C_ETC,   C_ETC,   C_ETC,   C_ETC,   C_LOW_L, C_ETC,   C_LOW_N, C_ETC,
    C_ETC,   C_ETC,   C_LOW_R, C_LOW_S, C_LOW_T, C_LOW_U, C_ETC,   C_ETC,
    C_ETC,   C_ETC,   C_ETC,   C_LCURB, C_ETC,   C_RCURB, C_ETC,   C_ETC
};


/*
    The state codes.
*/
enum states {
    GO,  /* start    */
    OK,  /* ok       */
    OB,  /* object   */
    KE,  /* key      */
    CO,  /* colon    */
    VA,  /* value    */
    AR,  /* array    */
    ST,  /* string   */
    ES,  /* escape   */
    U1,  /* u1       */
    U2,  /* u2       */
    U3,  /* u3       */
    U4,  /* u4       */
    MI,  /* minus    */
    ZE,  /* zero     */
    IN,  /* integer  */
    FR,  /* fraction */
    E1,  /* e        */
    E2,  /* ex       */
    E3,  /* exp      */
    T1,  /* tr       */
    T2,  /* tru      */
    T3,  /* true     */
    F1,  /* fa       */
    F2,  /* fal      */
    F3,  /* fals     */
    F4,  /* false    */
    N1,  /* nu       */
    N2,  /* nul      */
    N3,  /* null     */
    NO,  /* next object */
    NR_STATES
};


static int state_transition_table[NR_STATES][NR_CLASSES] = {
/*
    The state transition table takes the current state and the current symbol,
    and returns either a new state or an action. An action is represented as a
    negative number. A JSON text is accepted if at the end of the text the
    state is OK and if the mode is MODE_DONE.

              newline white                                                    1-9  ABCDF  etc
             space   |  |   {   }   [   ]   :   ,   "   \   /   +   -   .   0   |   a   b   c   d   e   f   l   n   r   s   t   u   |   E   |  */
/*start  GO*/ { GO, __, GO, -6, __, -5, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*ok     OK*/ { OK, NO, OK, __, -8, __, -7, __, -3, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*object OB*/ { OB, __, OB, __, -9, __, __, __, __, ST, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*key    KE*/ { KE, __, KE, __, __, __, __, __, __, ST, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*colon  CO*/ { CO, __, CO, __, __, __, __, -2, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*value  VA*/ { VA, VA, VA, -6, __, -5, __, __, __,-11, __, __, __, MI, __,-16,-10, __, __, __, __, __,-14, __,-14, __, __,-14, __, __, __, __},
/*array  AR*/ { AR, __, AR, -6, __, -5, -7, __, __,-11, __, __, __, MI, __,-16,-10, __, __, __, __, __,-14, __,-14, __, __,-14, __, __, __, __},
/*string ST*/ { ST, __, __, ST, ST, ST, ST, ST, ST, -4, ES, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST, ST},
/*escape ES*/ { __, __, __, __, __, __, __, __, __, ST, ST, ST, __, __, __, __, __, __, ST, __, __, __, ST, __, ST, ST, __, ST, U1, __, __, __},
/*u1     U1*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, U2, U2, U2, U2, U2, U2, U2, U2, __, __, __, __, __, __, U2, U2, __},
/*u2     U2*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, U3, U3, U3, U3, U3, U3, U3, U3, __, __, __, __, __, __, U3, U3, __},
/*u3     U3*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, U4, U4, U4, U4, U4, U4, U4, U4, __, __, __, __, __, __, U4, U4, __},
/*u4     U4*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, ST, ST, ST, ST, ST, ST, ST, ST, __, __, __, __, __, __, ST, ST, __},
/*minus  MI*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,-16, IN, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*zero   ZE*/ { OK, __, OK, __, -8, __, -7, __, -3, __, __, __, __, __, FR, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*int    IN*/ { OK, __, OK, __, -8, __, -7, __, -3, __, __, __, __, __, FR, IN, IN, __, __, __, __, E1, __, __, __, __, __, __, __, __, E1, __},
/*frac   FR*/ { OK, __, OK, __, -8, __, -7, __, -3, __, __, __, __, __, __, FR, FR, __, __, __, __, E1, __, __, __, __, __, __, __, __, E1, __},
/*e      E1*/ { __, __, __, __, __, __, __, __, __, __, __, __, E2, E2, __, E3, E3, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*ex     E2*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, E3, E3, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*exp    E3*/ { OK, __, OK, __, -8, __, -7, __, -3, __, __, __, __, __, __, E3, E3, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*tr     T1*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, T2, __, __, __, __, __, __},
/*tru    T2*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, T3, __, __, __},
/*true   T3*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,-15, __, __, __, __, __, __, __, __, __, __},
/*fa     F1*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, F2, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
/*fal    F2*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, F3, __, __, __, __, __, __, __, __},
/*fals   F3*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, F4, __, __, __, __, __},
/*false  F4*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,-15, __, __, __, __, __, __, __, __, __, __},
/*nu     N1*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, N2, __, __, __},
/*nul    N2*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, N3, __, __, __, __, __, __, __, __},
/*null   N3*/ { __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __,-15, __, __, __, __, __, __, __, __},
/*next   NO*/ { __, NO, __, -6, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __},
};


/*
    These modes can be pushed on the stack.
*/
enum modes {
    MODE_ARRAY, 
    MODE_DONE,  
    MODE_KEY,   
    MODE_OBJECT,
};

typedef struct JSON_checker_struct {
    int state;
    int depth;
    int top;
    int* stack;
} * JSON_checker;


extern JSON_checker new_JSON_checker(int depth);
extern int  JSON_checker_char(JSON_checker jc, int next_char);
extern int  JSON_checker_done(JSON_checker jc);

#endif
