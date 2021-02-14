# FinalProjectTelegramBot
Данный бот способен осуществлять технологию переноса стиля с одной фотографии на другую. Бот находится по адресу: @zendeer_style_bot. Имеется возможность запуска бота как на локальной машине, так и на сервере Heroku. Перед тем, как проверять работу на сервере просьба написать разработчику в телеграм по адресу @zendeer для уточнения функционирования бота на сервере.
Бот способен переносить стиль посредством реализации технологии: Style Transfer. За счет нее можно переносить стиль тремя разными способами:
1) Перенос стиля с одной картинки;
2) Перенос стиля с двух картинок стиля на разные части таргетной картинки;
3) Перенос смешанного стиля с двух картинок.


Бот способен работать в ассинхронном режиме за счет реализации фреймворка asyncio. 
Общение с ботом проходит в дружественном режиме. У пользователя есть возможность выбора периода ожидания обработки фотографии - либо за относительно небольшое время, но со стандартным качеством картинки, либо за весьма длительный промежуток, но с высоким качеством картинки.
