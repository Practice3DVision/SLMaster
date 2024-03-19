/**
 * @file typeDef.h
 * @author Evans Liu (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __TYPE_DEF_H_
#define __TYPE_DEF_H_

#define Q_PROPERTY_AUTO(TYPE, M)                                                                                       \
Q_PROPERTY(TYPE M MEMBER M##_ NOTIFY M##Changed)                                                                   \
    public:                                                                                                                \
    Q_SIGNAL void M##Changed();                                                                                        \
    void M(TYPE in_##M)                                                                                                \
{                                                                                                                  \
        M##_ = in_##M;                                                                                                 \
        Q_EMIT M##Changed();                                                                                           \
}                                                                                                                  \
    TYPE M()                                                                                                           \
{                                                                                                                  \
    return M##_;                                                                                                   \
}                                                                                                                  \
    private:                                                                                                               \
    TYPE M##_;

#endif// !__TYPE_DEF_H_
